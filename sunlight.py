"""A year of real sunlight at one bus: in how many daylight hours would a
solar project there need curtailing?

Uses results/sunlight_2024.csv (from fetch_sunlight.py: NASA POWER
satellite irradiance for 2024). Nothing here downloads anything.

For every daylight hour:
  sun share s     = that hour's irradiance / the year's brightest hour
  project output  = size x s
  other generators = 0.8 x s   (treated as solar, following the same sun)
  load            = 10%        (the worst case in decisions.md)
The ML model flags the hours that would break a limit; the real power flow
then rechecks each flagged hour. Hours the model passes are not rechecked,
so the solver count can miss the model's rare false passes (about 1% of
test cases, see results/model_report.md).
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
    would break a limit: first with the ML model, then rechecked with the
    real power flow.

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

    # Recheck each flagged hour with the real power flow. The project is
    # added once; each hour only changes the levels and solves again.
    net = add_project(load_grid(), bus, size_mw)
    project = net.sgen.index[-1]
    existing = net.sgen.index[:-1]
    net.load["scaling"] = LOAD_LEVEL
    confirmed, problems = [], []
    for i in np.flatnonzero(model_fails):
        net.sgen.loc[existing, "scaling"] = GEN_PEAK * s[i]
        net.sgen.loc[project, "p_mw"] = size_mw * s[i]
        result = check(net)
        if not result["passes"]:
            confirmed.append(sunlight.time_utc[i])
            problems.extend(result["problems"])

    return {
        "bus": int(bus),
        "size_mw": float(size_mw),
        "daylight_hours": int(len(sunlight)),
        "model_fail_hours": int(model_fails.sum()),
        "solver_fail_hours": len(confirmed),
        "fail_times_utc": confirmed,
        "limits": pd.Series([p.split(" ")[0] if p.startswith("voltage") else p.split(" is ")[0]
                             for p in problems], dtype=str).value_counts().to_dict(),
    }
