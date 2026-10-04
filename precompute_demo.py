"""Precompute every map the demo shows at its starting bus, so the demo
never waits on the solver.

Run it from the project folder:

    py -3.14 precompute_demo.py

For the demo bus (the farthest from a substation) it solves every slider
value, 0 to 20 MW in 0.1 MW steps, under each grid condition in the app,
and saves the results the maps and the pass/fail check need to
results/demo_cache.joblib. The app loads this file at startup; any other
bus is still solved live.
"""
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from engine import add_project, check, farthest_bus, load_grid, solve

OUT_FILE = Path(__file__).parent / "results" / "demo_cache.joblib"
# The same conditions as app.py: (load level, existing generation level),
# or None for the grid as pandapower ships it.
CONDITIONS = {
    "Today's grid": None,
    "Heavy load, other generators off": (0.6, 0.0),
}


def main():
    net = load_grid()
    solve(net)
    bus = farthest_bus(net)
    sizes = np.round(np.arange(0, 20.05, 0.1), 1)
    results = {}
    for condition, levels in CONDITIONS.items():
        base = load_grid()
        if levels is not None:
            base.load["scaling"], base.sgen["scaling"] = levels
        for mw in tqdm(sizes, desc=condition, unit="size"):
            project = add_project(base, bus, float(mw))
            result = check(project)
            solved = result["max_voltage"] is not None
            results[(condition, float(mw))] = {
                "result": result,
                # What the maps read; None if the power flow did not solve.
                "vm_pu": project.res_bus.vm_pu.to_numpy(np.float32) if solved else None,
                "line_loading": project.res_line.loading_percent.to_numpy(np.float32) if solved else None,
                "line_p_from_mw": project.res_line.p_from_mw.to_numpy(np.float32) if solved else None,
                "trafo_loading": project.res_trafo.loading_percent.to_numpy(np.float32) if solved else None,
            }
    OUT_FILE.parent.mkdir(exist_ok=True)
    joblib.dump({"bus": int(bus), "results": results}, OUT_FILE)
    print(f"Saved {len(results)} solved scenarios for bus {bus} to {OUT_FILE}.")


if __name__ == "__main__":
    main()
