"""Build fixed facts about each 20 kV bus, used as ML model inputs.

Run it from the project folder:

    py -3.14 build_features.py

It writes results/bus_features.csv, one row per bus. The model combines a
bus's row with a scenario's solar size, load level and generation level.
Nothing here needs a power flow per scenario, so the model stays fast.
"""
import copy
import math
from pathlib import Path

import networkx as nx
import pandas as pd
import pandapower.shortcircuit as sc
import pandapower.topology as top

from engine import load_grid

BUS_FEATURE_FILE = Path(__file__).parent / "results" / "bus_features.csv"
VN_KV = 20.0


def bus_paths(net):
    """Each 20 kV bus's path back to its substation.

    Returns {bus: dict} with the substation number (0 or 1), the feeder (the
    first bus out of the substation on the way to this bus), the lines on
    the path, and the path's resistance, reactance (ohms) and length (km).
    """
    graph = top.create_nxgraph(net, calc_branch_impedances=True)
    substations = list(net.trafo.lv_bus)
    facts = {}
    for bus in net.bus.index[net.bus.vn_kv == VN_KV]:
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
            "r_ohm": r,
            "x_ohm": x,
            "km": km,
        }
    return facts


def equivalent_impedance(net):
    """Each bus's equivalent (Thevenin) impedance, rk and xk in ohms, from
    a maximum short-circuit calculation.

    The grid lacks some short-circuit data, so this works on a copy with
    the assumptions in decisions.md: substation grid connection
    s_sc_max_mva = 1000 and rx_max = 0.1, generator k = 1.2.
    """
    sc_net = copy.deepcopy(net)
    sc_net.ext_grid["s_sc_max_mva"] = 1000
    sc_net.ext_grid["rx_max"] = 0.1
    sc_net.sgen["k"] = 1.2
    sc.calc_sc(sc_net, case="max")
    return sc_net.res_bus_sc[["rk_ohm", "xk_ohm"]]


def bus_features(net):
    """One row of fixed facts per 20 kV bus."""
    facts = bus_paths(net)
    impedance = equivalent_impedance(net)
    # A line's rating in MVA at 20 kV: sqrt(3) x voltage x current limit.
    rating_mva = math.sqrt(3) * VN_KV * net.line.max_i_ka

    rows = []
    for bus, f in facts.items():
        path_ratings = rating_mva[f["path_lines"]]
        rows.append({
            "bus": bus,
            "substation": f["substation"],
            "feeder": f["feeder"],
            "r_ohm": round(f["r_ohm"], 4),
            "x_ohm": round(f["x_ohm"], 4),
            "km": round(f["km"], 3),
            "rk_ohm": round(impedance.rk_ohm[bus], 4),
            "xk_ohm": round(impedance.xk_ohm[bus], 4),
            "n_lines": len(f["path_lines"]),
            # The weakest line between the bus and its substation; a bus at
            # the substation has no lines, so use the strongest line size.
            "path_min_rating_mva": round(path_ratings.min() if len(path_ratings)
                                         else rating_mva.max(), 2),
        })
    table = pd.DataFrame(rows).set_index("bus")

    # Rated existing generation and load per feeder (MW), so the model
    # knows what else shares the feeder.
    for column, element in [("feeder_sgen_mw", net.sgen), ("feeder_load_mw", net.load)]:
        per_feeder = element.p_mw.groupby(element.bus.map(table.feeder)).sum()
        table[column] = table.feeder.map(per_feeder).fillna(0).round(2)
    return table


def main():
    table = bus_features(load_grid())
    BUS_FEATURE_FILE.parent.mkdir(exist_ok=True)
    table.to_csv(BUS_FEATURE_FILE)
    print(f"Saved {len(table)} buses to {BUS_FEATURE_FILE}.")


if __name__ == "__main__":
    main()
