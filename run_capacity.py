"""Run find_capacity for every 20 kV bus and save the results.

Run it from the project folder:

    py -3.14 run_capacity.py

It writes results/capacity.csv, which the app's Capacity view reads.
"""
import time

import pandas as pd
from tqdm import tqdm

from capacity import find_capacity
from capacity_data import CAPACITY_FILE
from engine import load_grid


def main():
    net = load_grid()
    mv_buses = list(net.bus.index[net.bus.vn_kv == 20])

    rows = []
    for bus in tqdm(mv_buses, desc="Finding capacity", unit="bus"):
        start = time.perf_counter()
        capacity, limit = find_capacity(net, bus)
        run_time = time.perf_counter() - start
        rows.append({
            "bus": int(bus),
            "capacity_mw": round(capacity, 1),
            # find_capacity gives a list of problems (or None): save as text.
            "limit": "; ".join(limit) if limit else "",
            "run_time_s": round(run_time, 2),
        })

    # Write to a temporary file first, then swap it in, so the app never
    # reads a half-written file.
    CAPACITY_FILE.parent.mkdir(exist_ok=True)
    temp_file = CAPACITY_FILE.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temp_file, index=False)
    temp_file.replace(CAPACITY_FILE)

    total = sum(row["run_time_s"] for row in rows)
    print(f"Saved {len(rows)} buses to {CAPACITY_FILE} in {total:.0f} s.")


if __name__ == "__main__":
    main()
