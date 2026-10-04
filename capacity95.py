"""The 95% capacity map: the solar size each bus can take in at least 95%
of grid conditions.

Two stages, run from the project folder (after train_model.py):

    py -3.14 capacity95.py              stage 1: the model
    py -3.14 capacity95.py --recheck    stage 2: the real solver

Stage 1: sample 1,000 grid conditions over the ranges in decisions.md. The
model finds each bus's capacity under every condition (bisection, no power
flow); the 95% capacity is the size that fits in at least 95% of them.
Saves results/capacity95.csv.

Stage 2: recheck each bus's 95% capacity with the real power flow on 50
new conditions. If more than 5% fail, lower it by 0.1 MW and recheck, until
it passes. Also solves 0.1 MW above the final size to record the limit that
stops a bigger project. Updates results/capacity95.csv and writes
results/capacity95_recheck.json (how many buses were lowered).
"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import ml
from capacity import MAX_SEARCH_MW
from make_scenarios import DEFAULT_WORKERS, _start_worker, solve_chunk

ROOT = Path(__file__).parent
OUT_FILE = ROOT / "results" / "capacity95.csv"
SUMMARY_FILE = ROOT / "results" / "capacity95_recheck.json"
N_CONDITIONS = 1000
N_RECHECK = 50
CONFIDENCE = 0.95
MODEL_SEED = 21
RECHECK_SEED = 22  # a different seed: the recheck uses new conditions
STEP_MW = 0.1


def sample_conditions(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "load_level": rng.uniform(0.10, 0.60, n).round(3),
        "gen_level": rng.uniform(0.0, 0.80, n).round(3),
    })


def confident_size(capacities, confidence=CONFIDENCE):
    """The largest size that fits in at least `confidence` of conditions,
    given each condition's capacity."""
    ordered = np.sort(np.asarray(capacities))
    allowed_failures = int(np.floor((1 - confidence) * len(ordered)))
    return float(ordered[allowed_failures])


def model_limit(models, bus_features, bus, mw, conditions):
    """What the model says breaks first just above `mw` (voltage, a line or
    a transformer), most often across the conditions."""
    x = conditions.assign(bus=bus, solar_mw=round(mw + STEP_MW, 1))
    p = ml.predict(models, ml.add_features(x, bus_features))
    kinds = Counter()
    for _, row in p.iterrows():
        if row.max_voltage_pu > ml.MAX_VOLTAGE_PU:
            kinds["voltage (model)"] += 1
        if row.max_line_loading > ml.MAX_LOADING_PERCENT:
            kinds["a line (model)"] += 1
        if row.max_trafo_loading > ml.MAX_LOADING_PERCENT:
            kinds["a transformer (model)"] += 1
    return kinds.most_common(1)[0][0] if kinds else ""


def stage_model():
    conditions = sample_conditions(N_CONDITIONS, MODEL_SEED)
    models, bus_features = ml.load_model(), ml.read_bus_features()
    buses = list(bus_features.index)
    cases = pd.DataFrame({
        "bus": np.repeat(buses, len(conditions)),
        "load_level": np.tile(conditions.load_level.to_numpy(), len(buses)),
        "gen_level": np.tile(conditions.gen_level.to_numpy(), len(buses)),
    })
    print(f"Model: {len(cases):,} bus-and-condition capacities (bisection)...")
    cases["capacity"] = ml.model_capacity_bisect(models, bus_features, cases).to_numpy()

    rows = []
    for bus, group in cases.groupby("bus"):
        size = confident_size(group.capacity)
        rows.append({
            "bus": bus,
            "model95_mw": size,
            "capacity95_mw": size,  # replaced by the rechecked value in stage 2
            # share of the 1,000 conditions where the model says `size` fits
            "pass_share": round(float((group.capacity >= size - 1e-9).mean()), 3),
            "checked_by": "model, 1,000 conditions",
            "lowered_by_mw": 0.0,
            "limit": "" if size >= MAX_SEARCH_MW else
                     model_limit(models, bus_features, bus, size, conditions),
        })
    OUT_FILE.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
    print(f"Saved {len(rows)} buses to {OUT_FILE}.")


def solve_all(pool, sizes, conditions, label):
    """Solve every bus in `sizes` (bus -> MW) under every condition.
    Returns {bus: (share that fail, list of problem texts)}."""
    rows = [{"scenario": i, "bus": bus, "solar_mw": mw,
             "load_level": c.load_level, "gen_level": c.gen_level}
            for bus, mw in sizes.items() for i, c in conditions.iterrows()]
    chunks = [rows[i:i + 100] for i in range(0, len(rows), 100)]
    results = [r for chunk in tqdm(pool.map(solve_chunk, chunks), total=len(chunks),
                                   desc=label, unit="chunk")
               for r in chunk]
    table = pd.DataFrame(results)
    return {bus: (1 - g.passes.mean(), g.problems[~g.passes].tolist())
            for bus, g in table.groupby("bus")}


def solver_limit(problems):
    """The most common thing that broke, e.g. "voltage" or "Line 27"."""
    kinds = Counter()
    for text in problems:
        for p in text.split("; "):
            kinds["voltage" if p.startswith("voltage") else
                  "did not solve" if p == "did not solve" else
                  p.split(" is loaded")[0]] += 1
    return kinds.most_common(1)[0][0] if kinds else ""


def stage_recheck(workers):
    table = pd.read_csv(OUT_FILE, keep_default_na=False).set_index("bus")
    conditions = sample_conditions(N_RECHECK, RECHECK_SEED)
    size = table.model95_mw.to_dict()
    fail_share = {}
    lowered = Counter()
    todo = dict(size)
    with ProcessPoolExecutor(max_workers=workers, initializer=_start_worker) as pool:
        round_number = 0
        while todo:
            round_number += 1
            results = solve_all(pool, todo, conditions,
                                f"Recheck round {round_number}: {len(todo)} buses")
            next_todo = {}
            for bus, mw in todo.items():
                share, _ = results[bus]
                fail_share[bus] = share
                if share > 1 - CONFIDENCE and mw > 0:
                    size[bus] = round(max(mw - STEP_MW, 0.0), 1)
                    lowered[bus] += 1
                    next_todo[bus] = size[bus]
            todo = next_todo

        # The limit that stops a bigger project: solve 0.1 MW above each final size.
        above = {bus: round(mw + STEP_MW, 1) for bus, mw in size.items()
                 if mw < MAX_SEARCH_MW}
        above_results = solve_all(pool, above, conditions, "Limits just above")

    for bus in table.index:
        table.loc[bus, "capacity95_mw"] = size[bus]
        table.loc[bus, "pass_share"] = round(1 - fail_share[bus], 3)
        table.loc[bus, "checked_by"] = "solver, 50 new conditions"
        table.loc[bus, "lowered_by_mw"] = round(lowered[bus] * STEP_MW, 1)
        if bus in above_results and above_results[bus][1]:
            table.loc[bus, "limit"] = solver_limit(above_results[bus][1])
        elif bus in above:
            # 0.1 MW more still passes on these 50 conditions; keep the model's word.
            table.loc[bus, "limit"] = table.limit[bus]
        else:
            table.loc[bus, "limit"] = ""
    table.reset_index().to_csv(OUT_FILE, index=False)

    summary = {
        "buses": int(len(table)),
        "buses_lowered": int(sum(1 for b in lowered if lowered[b])),
        "total_lowered_mw": round(sum(lowered.values()) * STEP_MW, 1),
        "most_lowered_mw": round(max(lowered.values(), default=0) * STEP_MW, 1),
        "recheck_rounds": round_number,
        "recheck_conditions": N_RECHECK,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    print(f"Rechecked {len(table)} buses; lowered {summary['buses_lowered']}. "
          f"Saved {OUT_FILE} and {SUMMARY_FILE}.")


if __name__ == "__main__":
    if "--recheck" in sys.argv:
        workers = DEFAULT_WORKERS
        if "--workers" in sys.argv:
            workers = int(sys.argv[sys.argv.index("--workers") + 1])
        stage_recheck(workers)
    else:
        stage_model()
