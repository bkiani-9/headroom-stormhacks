"""Build the ML model's inputs (features) for every (condition, bus).

Run it from the project folder:

    py -3.14 build_features.py

It writes results/features.csv, one row per (condition, bus), matching the
rows of results/scenarios.csv. It never runs the capacity search: only one
ordinary power flow per condition (see PLAN.md, ML design).
"""
from pathlib import Path

import networkx as nx
import pandas as pd
import pandapower.topology as top

from engine import load_grid, solve
from run_scenarios import grid_at, make_conditions

FEATURE_FILE = Path(__file__).parent / "results" / "features.csv"


def bus_paths(net):
    """Fixed facts about each 20 kV bus: its path back to its substation.

    Returns {bus: dict} with the substation number (0 or 1), the feeder (the
    first bus out of the substation on the way to this bus), the lines on
    the path, and the path's resistance, reactance (ohms) and length (km).
    """
    graph = top.create_nxgraph(net, calc_branch_impedances=True)
    substations = list(net.trafo.lv_bus)
    facts = {}
    for bus in net.bus.index[net.bus.vn_kv == 20]:
        # The grid is in two separate halves: find the substation that feeds this bus.
        sub_number, sub_bus = next(
            (i, s) for i, s in enumerate(substations) if nx.has_path(graph, s, bus)
        )
        path = nx.shortest_path(graph, sub_bus, bus, weight="z_ohm")
        r = x = km = 0.0
        lines = []
        for a, b in zip(path, path[1:]):
            # Pick the line between a and b (the lowest-impedance one if
            # there are several in parallel).
            key, data = min(
                ((k, d) for k, d in graph.get_edge_data(a, b).items() if k[0] == "line"),
                key=lambda item: item[1]["z_ohm"],
            )
            lines.append(int(key[1]))
            r += data["r_ohm"]
            x += data["x_ohm"]
            km += data["weight"]
        facts[int(bus)] = {
            "substation": sub_number,
            "feeder": int(path[1]) if len(path) > 1 else int(bus),
            "path_lines": lines,
            "r_ohm": round(r, 4),
            "x_ohm": round(x, 4),
            "km": round(km, 3),
        }
    return facts


def features_for(condition, facts):
    """One row per bus for one condition, from one ordinary power flow."""
    net = grid_at(condition)
    solve(net)
    vm = net.res_bus.vm_pu
    loading = net.res_line.loading_percent

    feeder_max_vm = {}
    for bus, f in facts.items():
        feeder_max_vm[f["feeder"]] = max(feeder_max_vm.get(f["feeder"], 0), vm[bus])

    rows = []
    for bus, f in facts.items():
        rows.append({
            "condition": condition["condition"],
            "bus": bus,
            "load_level": condition["load_level"],
            "gen_level": condition["gen_level"],
            "substation": f["substation"],
            "r_ohm": f["r_ohm"],
            "x_ohm": f["x_ohm"],
            "km": f["km"],
            "bus_vm_pu": round(vm[bus], 5),
            "feeder_max_vm_pu": round(feeder_max_vm[f["feeder"]], 5),
            "path_max_loading": round(max(loading[f["path_lines"]]), 2)
                                if f["path_lines"] else 0.0,
        })
    return rows


def main():
    facts = bus_paths(load_grid())
    rows = []
    for condition in make_conditions():
        rows.extend(features_for(condition, facts))
    FEATURE_FILE.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(FEATURE_FILE, index=False)
    print(f"Saved {len(rows)} rows to {FEATURE_FILE}.")


if __name__ == "__main__":
    main()
