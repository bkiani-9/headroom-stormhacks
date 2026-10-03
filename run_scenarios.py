"""Run find_capacity at every 20 kV bus under many grid conditions.

This makes the training data for the ML model (see PLAN.md, ML design).
Run it from the project folder:

    py -3.14 run_scenarios.py

It writes results/capacity_scenarios.csv, one row per (condition, bus). Results are
saved after each condition, and conditions already in the file are skipped,
so if it stops, run it again to carry on.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm

from capacity import find_capacity
from engine import load_grid

SCENARIO_FILE = Path(__file__).parent / "results" / "capacity_scenarios.csv"

# Ranges from decisions.md, as fractions of rated size.
LOAD_RANGE = (0.10, 0.60)
GEN_RANGE = (0.0, 0.80)
N_SAMPLED = 40
SEED = 42


def make_conditions():
    """The four corners, then N_SAMPLED conditions spread evenly over the
    load x generation square (Latin hypercube, fixed seed)."""
    corners = [(load, gen) for load in LOAD_RANGE for gen in GEN_RANGE]
    sampler = qmc.LatinHypercube(d=2, rng=np.random.default_rng(SEED))
    points = qmc.scale(sampler.random(N_SAMPLED),
                       [LOAD_RANGE[0], GEN_RANGE[0]],
                       [LOAD_RANGE[1], GEN_RANGE[1]])
    sampled = [(round(load, 3), round(gen, 3)) for load, gen in points]
    conditions = []
    for i, (load, gen) in enumerate(corners + sampled):
        conditions.append({"condition": i, "load_level": load,
                           "gen_level": gen, "corner": i < len(corners)})
    return conditions


def limit_type(limit):
    """Sort the limit text into voltage / line / transformer / none."""
    if not limit:
        return "none"
    text = "; ".join(limit)
    if text == "did not solve":
        return "did not solve"
    if text.startswith("voltage"):
        return "voltage"
    if "Transformer" in text:
        return "transformer"
    return "line"


def grid_at(condition):
    """The grid with every load and existing generator scaled."""
    net = load_grid()
    net.load["scaling"] = condition["load_level"]
    net.sgen["scaling"] = condition["gen_level"]
    return net


def run(conditions, buses=None, out_file=SCENARIO_FILE):
    done = set()
    if out_file.exists():
        done = set(pd.read_csv(out_file, usecols=["condition"]).condition)
    todo = [c for c in conditions if c["condition"] not in done]
    print(f"{len(done)} conditions already saved, {len(todo)} to run.")

    out_file.parent.mkdir(exist_ok=True)
    for condition in tqdm(todo, desc="Conditions", unit="cond"):
        net = grid_at(condition)
        bus_list = buses if buses is not None else list(net.bus.index[net.bus.vn_kv == 20])
        rows = []
        for bus in bus_list:
            start = time.perf_counter()
            capacity, limit = find_capacity(net, bus)
            rows.append({
                **condition,
                "bus": int(bus),
                "capacity_mw": round(capacity, 1),
                "limit_type": limit_type(limit),
                "limit": "; ".join(limit) if limit else "",
                "run_time_s": round(time.perf_counter() - start, 2),
            })
        # Append this condition's rows; write the header only the first time.
        pd.DataFrame(rows).to_csv(out_file, mode="a", index=False,
                                  header=not out_file.exists())


if __name__ == "__main__":
    run(make_conditions())
