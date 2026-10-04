"""Solve 20,000 random scenarios with the real power flow, using all cores.

Each scenario: a random 20 kV bus, a random solar size (0-20 MW), and a
random grid condition (load 10-60%, existing generation 0-80%, ranges
from decisions.md). Run it from the project folder:

    py -3.14 make_scenarios.py

It writes results/scenarios.csv, one row per scenario.

    py -3.14 make_scenarios.py --near-boundary

makes a second set, results/scenarios_boundary.csv: random buses and
conditions again, but with solar sizes within 2 MW of where the current
model thinks the capacity is, so training sees many cases close to the
pass/fail line. The real power flow still decides every result.

    py -3.14 make_scenarios.py --edges

makes a third set, results/scenarios_edges.csv: like --near-boundary, but
with load and generation levels drawn more often near the ends of their
ranges, where uniform sampling leaves few examples (e.g. the low-load,
high-generation "generation" case the app shows).
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from engine import add_project, check, load_grid

SCENARIO_FILE = Path(__file__).parent / "results" / "scenarios.csv"
BOUNDARY_FILE = Path(__file__).parent / "results" / "scenarios_boundary.csv"
EDGE_FILE = Path(__file__).parent / "results" / "scenarios_edges.csv"
N_SCENARIOS = 20_000
SEED = 7
BOUNDARY_SEED = 8
EDGE_SEED = 9
BOUNDARY_SPREAD_MW = 2.0
CHUNK = 250  # scenarios per job handed to a worker
# Worker processes. Each holds its own copy of the grid, so 8 ran the
# laptop out of memory; 4 is safe. Change with --workers N.
DEFAULT_WORKERS = 4

# Each worker process loads the grid once and reuses it.
_net = None


def _start_worker():
    global _net
    warnings.filterwarnings("ignore")  # pandapower's deprecation noise
    _net = load_grid()


def make_inputs(n=N_SCENARIOS, seed=SEED):
    """The random scenarios (inputs only), from a fixed seed."""
    rng = np.random.default_rng(seed)
    buses = load_grid().bus.query("vn_kv == 20").index.to_numpy()
    return pd.DataFrame({
        "scenario": np.arange(n),
        "bus": rng.choice(buses, n),
        "solar_mw": rng.uniform(0, 20, n).round(2),
        "load_level": rng.uniform(0.10, 0.60, n).round(3),
        "gen_level": rng.uniform(0.0, 0.80, n).round(3),
    })


def make_boundary_inputs(n=N_SCENARIOS, seed=BOUNDARY_SEED, edges=False):
    """Random buses and conditions, with sizes near the model's capacity.

    With `edges`, load and generation levels follow a Beta(0.5, 0.5)
    shape over their ranges: still anywhere in the range, but more often
    near its ends.
    """
    import ml  # only needed here, after a first model exists

    inputs = make_inputs(n, seed)
    rng = np.random.default_rng(seed + 1)
    if edges:
        inputs["load_level"] = (0.10 + 0.50 * rng.beta(0.5, 0.5, n)).round(3)
        inputs["gen_level"] = (0.80 * rng.beta(0.5, 0.5, n)).round(3)
    capacity = ml.model_capacity(ml.load_model(), ml.read_bus_features(), inputs)
    nudge = rng.uniform(-BOUNDARY_SPREAD_MW, BOUNDARY_SPREAD_MW, n)
    inputs["solar_mw"] = np.clip(capacity + nudge, 0, 20).round(2)
    return inputs


def solve_chunk(rows):
    """Solve a list of scenarios in this worker; return result rows."""
    out = []
    for s in rows:
        _net.load["scaling"] = s["load_level"]
        _net.sgen["scaling"] = s["gen_level"]
        result = check(add_project(_net, int(s["bus"]), s["solar_mw"]))
        out.append({
            **s,
            "max_voltage_pu": result["max_voltage"],
            "max_line_loading": result["max_line"],
            "max_trafo_loading": result["max_trafo"],
            "passes": result["passes"],
            "problems": "; ".join(result["problems"]),
        })
    return out


def main():
    if "--edges" in sys.argv:
        out_file, inputs = EDGE_FILE, make_boundary_inputs(seed=EDGE_SEED, edges=True)
    elif "--near-boundary" in sys.argv:
        out_file, inputs = BOUNDARY_FILE, make_boundary_inputs()
    else:
        out_file, inputs = SCENARIO_FILE, make_inputs()
    inputs = inputs.to_dict("records")
    chunks = [inputs[i:i + CHUNK] for i in range(0, len(inputs), CHUNK)]
    workers = DEFAULT_WORKERS
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    workers = min(workers, os.cpu_count())
    print(f"{len(inputs)} scenarios on {workers} processes.")

    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_start_worker) as pool:
        for chunk_rows in tqdm(pool.map(solve_chunk, chunks), total=len(chunks),
                               desc="Chunks", unit="chunk"):
            rows.extend(chunk_rows)

    # Write to a temporary file first, then swap it in.
    out_file.parent.mkdir(exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temp_file, index=False)
    temp_file.replace(out_file)
    print(f"Saved {len(rows)} scenarios to {out_file}.")


# Needed on Windows: worker processes re-import this file, and this guard
# stops each of them starting its own pool.
if __name__ == "__main__":
    main()
