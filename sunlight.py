"""A year of real sunlight at one bus: in how many daylight hours would a
solar project there need curtailing?

Uses results/sunlight_2024.csv (from fetch_sunlight.py: NASA POWER
satellite irradiance for 2024). Nothing here downloads anything.

For every daylight hour:
  sun share s     = that hour's irradiance / the year's brightest hour
  project output  = size x s
  other generators = 0.8 x s   (treated as solar, following the same sun)
  load            = 10%        (the worst case in decisions.md)
With demand held at its minimum, more sunlight only pushes voltages and
reverse power flows up, so the hours that break a limit are exactly the
hours above one level of sunlight. The real power flow finds that level by
halving (about 14 solves, instead of one per hour), and every hour is then
sorted by it. The ML model's own count is kept alongside, for comparison.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import ml
from engine import add_project, check, load_grid

SUNLIGHT_FILE = Path(__file__).parent / "results" / "sunlight_2024.csv"
LOAD_LEVEL = 0.10   # worst case for voltage
GEN_PEAK = 0.80     # existing generators at the brightest hour


def read_sunlight(path=SUNLIGHT_FILE):
    """Daylight hours only, with the sun share s (0 to 1) of each."""
    table = pd.read_csv(path, parse_dates=["time_utc"])
    table = table[table.irradiance_w_m2 > 0].copy()
    table["sun_share"] = table.irradiance_w_m2 / table.irradiance_w_m2.max()
    return table.reset_index(drop=True)


def year_check(bus, size_mw, models=None, bus_features=None, sunlight=None):
    """Count the daylight hours in 2024 where `size_mw` of solar at `bus`
    would break a limit: with the ML model, and with the real power flow
    (by finding the sun share where a limit first breaks).

    Returns a dict: daylight_hours, model_fail_hours, solver_fail_hours,
    the times of the solver-confirmed hours, and the limits that broke.
    """
    sunlight = read_sunlight() if sunlight is None else sunlight
    models = ml.load_model() if models is None else models
    bus_features = ml.read_bus_features() if bus_features is None else bus_features

    s = sunlight.sun_share.to_numpy()
    cases = pd.DataFrame({
        "bus": bus,
        "solar_mw": size_mw * s,
        "load_level": LOAD_LEVEL,
        "gen_level": GEN_PEAK * s,
    })
    model_fails = ~ml.passes(ml.predict(models, ml.add_features(cases, bus_features))).to_numpy()

    # The real power flow, at a given sun share: the project and the other
    # generators follow the sun, demand stays at its minimum.
    net = add_project(load_grid(), bus, size_mw)
    project = net.sgen.index[-1]
    existing = net.sgen.index[:-1]
    net.load["scaling"] = LOAD_LEVEL

    def solve_at(share):
        net.sgen.loc[existing, "scaling"] = GEN_PEAK * share
        net.sgen.loc[project, "p_mw"] = size_mw * share
        return check(net)

    # Find the sun share where a limit first breaks, by halving between a
    # share that passes and one that fails, to 1/8192 of the brightest hour.
    fails = np.zeros(len(s), dtype=bool)
    problems = []
    top = solve_at(float(s.max()))
    if not top["passes"]:
        problems = top["problems"]
        bottom = solve_at(0.0)
        if not bottom["passes"]:
            fails[:] = True
        else:
            low, high = 0.0, float(s.max())
            while high - low > 1 / 8192:
                mid = (low + high) / 2
                result = solve_at(mid)
                if result["passes"]:
                    low = mid
                else:
                    high, problems = mid, result["problems"]
            fails = s >= high
            # The few hours between the last pass and the last fail: solve each.
            for i in np.flatnonzero((s > low) & (s < high)):
                fails[i] = not solve_at(float(s[i]))["passes"]
    confirmed = list(sunlight.time_utc[fails])

    return {
        "bus": int(bus),
        "size_mw": float(size_mw),
        "daylight_hours": int(len(sunlight)),
        "model_fail_hours": int(model_fails.sum()),
        "solver_fail_hours": len(confirmed),
        "fail_times_utc": confirmed,
        "limits": [p.split(" ")[0] if p.startswith("voltage") else p.split(" is ")[0]
                   for p in problems],
    }
