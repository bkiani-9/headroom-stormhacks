import pandapower as pp
import pandapower.networks as pn

# Limits from decisions.md
MAX_VOLTAGE_PU = 1.04
MAX_LOADING_PERCENT = 100.0


def load_grid():
    """Load the MV Oberrhein grid (generation case), not yet solved."""
    return pn.mv_oberrhein("generation")


def check_limits(net):
    """Solve the grid and check it against the voltage and loading limits.

    Returns a dict with:
      passes   - True if every limit is respected
      problems - list of the limits that broke, in plain words
      plus the highest voltage, line loading and transformer loading.
    """
    pp.runpp(net, numba=False)

    max_voltage = net.res_bus.vm_pu.max()
    max_line = net.res_line.loading_percent.max()
    max_trafo = net.res_trafo.loading_percent.max()

    problems = []
    if max_voltage > MAX_VOLTAGE_PU:
        problems.append(f"voltage {max_voltage:.4f} pu is above {MAX_VOLTAGE_PU}")
    if max_line > MAX_LOADING_PERCENT:
        problems.append(f"a line is loaded to {max_line:.1f}%")
    if max_trafo > MAX_LOADING_PERCENT:
        problems.append(f"a transformer is loaded to {max_trafo:.1f}%")

    return {
        "passes": not problems,
        "problems": problems,
        "max_voltage": max_voltage,
        "max_line": max_line,
        "max_trafo": max_trafo,
    }
