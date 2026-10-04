"""The 95% capacity map: the solar size each bus can take in at least 95%
of grid conditions, found with the ML model and rechecked with the solver.

Run it from the project folder (after train_model.py):

    py -3.14 capacity95.py

1. Sample N_CONDITIONS grid conditions over the ranges in decisions.md.
2. The model gives each bus's capacity under every condition (fast, no
   power flow). The 95% capacity is the size that fits in at least 95%
   of them.
3. Physics recheck: solve every bus at that size under every condition
   with the real power flow. If fewer than 95% pass, step down 0.1 MW and
   recheck; if 0.1 MW more would still pass in 95%, step up. So the final
   number rests on the solver, not the model.

Writes results/capacity95.csv, one row per bus.
"""
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

OUT_FILE = Path(__file__).parent / "results" / "capacity95.csv"
N_CONDITIONS = 200
CONFIDENCE = 0.95
SEED = 11
STEP_MW = 0.1
MAX_ROUNDS = 40  # safety stop for the step-up/step-down loop


def sample_conditions(n=N_CONDITIONS, seed=SEED):
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


def solve_sizes(pool, sizes, conditions):
    """Solve every bus in `sizes` (bus -> MW) under every condition.
    Returns {bus: (share that pass, list of problem texts of failures)}."""
    rows = []
    for bus, mw in sizes.items():
        for i, c in conditions.iterrows():
            rows.append({"scenario": i, "bus": bus, "solar_mw": mw,
                         "load_level": c.load_level, "gen_level": c.gen_level})
    chunks = [rows[i:i + 200] for i in range(0, len(rows), 200)]
    results = [r for chunk in tqdm(pool.map(solve_chunk, chunks), total=len(chunks),
                                   desc=f"Solving {len(sizes)} buses", unit="chunk", leave=False)
               for r in chunk]
    out = {}
    table = pd.DataFrame(results)
    for bus, group in table.groupby("bus"):
        out[bus] = (group.passes.mean(), group.problems[~group.passes].tolist())
    return out


def limit_kind(problem):
    if problem == "did not solve":
        return "did not solve"
    if problem.startswith("voltage"):
        return "voltage"
    return problem.split(" is loaded")[0]  # e.g. "Line 27"


def main():
    workers = DEFAULT_WORKERS
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    conditions = sample_conditions()
    models, bus_features = ml.load_model(), ml.read_bus_features()
    buses = list(bus_features.index)

    # Step 2: the model's capacity for every bus under every condition.
    cases = pd.DataFrame([{"bus": b, "load_level": c.load_level, "gen_level": c.gen_level}
                          for b in buses for _, c in conditions.iterrows()])
    print(f"Model: {len(cases):,} bus-and-condition capacities...")
    cases["capacity"] = ml.model_capacity(models, bus_features, cases).to_numpy()
    model95 = cases.groupby("bus").capacity.apply(confident_size)

    # Step 3: physics recheck, stepping down or up 0.1 MW at a time.
    size = model95.round(1).to_dict()
    passed = {}       # bus -> (size known to pass in >= 95%, its pass share)
    failed = {}       # bus -> (smallest size known to fail, problems there)
    todo = dict(size)
    with ProcessPoolExecutor(max_workers=workers, initializer=_start_worker) as pool:
        for round_number in range(1, MAX_ROUNDS + 1):
            if not todo:
                break
            print(f"Recheck round {round_number}: {len(todo)} buses")
            results = solve_sizes(pool, todo, conditions)
            next_todo = {}
            for bus, mw in todo.items():
                share, problems = results[bus]
                if share >= CONFIDENCE:
                    passed[bus] = (mw, share)
                    up = round(mw + STEP_MW, 1)
                    # Step up unless we're at the top or already know `up` fails.
                    if up <= MAX_SEARCH_MW and not (bus in failed and failed[bus][0] <= up):
                        next_todo[bus] = up
                else:
                    failed[bus] = (mw, problems)
                    down = round(mw - STEP_MW, 1)
                    if bus not in passed and down >= 0:
                        next_todo[bus] = down
                    elif down < 0:
                        passed[bus] = (0.0, share)
            todo = next_todo

    rows = []
    for bus in buses:
        mw, share = passed.get(bus, (0.0, float("nan")))
        stop = ""
        if bus in failed:
            kinds = Counter(limit_kind(p) for text in failed[bus][1] for p in text.split("; "))
            stop = kinds.most_common(1)[0][0] if kinds else ""
        rows.append({
            "bus": bus,
            "capacity95_mw": mw,
            "pass_share": round(share, 3),
            "model95_mw": model95[bus],
            "limit": stop,
        })
    OUT_FILE.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
    table = pd.DataFrame(rows)
    print(f"Saved {len(rows)} buses to {OUT_FILE}.")
    print(f"Model vs rechecked 95% capacity: mean difference "
          f"{(table.model95_mw - table.capacity95_mw).abs().mean():.2f} MW")


if __name__ == "__main__":
    main()
