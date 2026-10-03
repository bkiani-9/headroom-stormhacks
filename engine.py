import copy

import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

# Limits from decisions.md
MAX_VOLTAGE_PU = 1.04
MAX_LOADING_PERCENT = 100.0


def load_grid():
    """Load the MV Oberrhein grid (generation case), not yet solved."""
    return pn.mv_oberrhein("generation")


def farthest_bus(net):
    """Return the 20 kV bus electrically farthest from its nearest substation.

    "Farthest" means the most line impedance (ohms) between the bus and
    the 20 kV side of the closest substation transformer.
    """
    graph = top.create_nxgraph(net, calc_branch_impedances=True)
    distances = pd.DataFrame({
        sub: top.calc_distance_to_bus(net, sub, g=graph, weight="z_ohm")
        for sub in net.trafo.lv_bus
    })
    nearest = distances.min(axis=1)
    mv_buses = net.bus.index[net.bus.vn_kv == 20]
    return int(nearest.reindex(mv_buses).idxmax())


def add_project(net, bus, mw):
    """Return a copy of `net` with a new generator of `mw` megawatts at `bus`.

    The original `net` is left untouched, so a cached grid stays clean.
    """
    new_net = copy.deepcopy(net)
    pp.create_sgen(new_net, bus, p_mw=mw, name="New project")
    return new_net


def solve(net):
    """Run the power flow. Results land in net.res_bus, net.res_line, etc."""
    pp.runpp(net, numba=False)


def check(net):
    """Solve the grid and check it against the voltage and loading limits.

    Returns a dict with:
      passes          - True if every limit is respected
      problems        - list of the limits that broke, in plain words
      max_voltage     - highest bus voltage (per unit)
      max_line        - loading of the busiest line (%)
      worst_line      - index of the busiest line
      worst_line_name - name of the busiest line
      max_trafo, worst_trafo, worst_trafo_name - the same for transformers
    If the power flow does not solve, passes is False, problems is
    ["did not solve"] and the numbers are None.
    """
    try:
        solve(net)
    except pp.LoadflowNotConverged:
        return {
            "passes": False,
            "problems": ["did not solve"],
            "max_voltage": None,
            "max_line": None,
            "worst_line": None,
            "worst_line_name": None,
            "max_trafo": None,
            "worst_trafo": None,
            "worst_trafo_name": None,
        }

    max_voltage = float(net.res_bus.vm_pu.max())

    worst_line = int(net.res_line.loading_percent.idxmax())
    max_line = float(net.res_line.loading_percent[worst_line])
    worst_line_name = net.line.name[worst_line]

    worst_trafo = int(net.res_trafo.loading_percent.idxmax())
    max_trafo = float(net.res_trafo.loading_percent[worst_trafo])
    worst_trafo_name = net.trafo.name[worst_trafo]

    problems = []
    if max_voltage > MAX_VOLTAGE_PU:
        problems.append(f"voltage {max_voltage:.4f} pu is above {MAX_VOLTAGE_PU}")
    if max_line > MAX_LOADING_PERCENT:
        problems.append(f"{worst_line_name} is loaded to {max_line:.1f}%")
    if max_trafo > MAX_LOADING_PERCENT:
        problems.append(f"{worst_trafo_name} is loaded to {max_trafo:.1f}%")

    return {
        "passes": not problems,
        "problems": problems,
        "max_voltage": max_voltage,
        "max_line": max_line,
        "worst_line": worst_line,
        "worst_line_name": worst_line_name,
        "max_trafo": max_trafo,
        "worst_trafo": worst_trafo,
        "worst_trafo_name": worst_trafo_name,
    }
