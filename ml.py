"""The ML model: shared feature recipe, loading, and predictions.

The model predicts the three numbers check() looks at (highest voltage,
busiest line loading, busiest transformer loading) for a solar project of
a given size at a bus under a given grid condition. Pass or fail then
comes from the same limits as engine.check(). No power flow is run.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from engine import MAX_LOADING_PERCENT, MAX_VOLTAGE_PU

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "models" / "headroom_model.joblib"
BUS_FEATURE_FILE = ROOT / "results" / "bus_features.csv"

TARGETS = ["max_voltage_pu", "max_line_loading", "max_trafo_loading"]
FEATURES = [
    "solar_mw", "load_level", "gen_level",
    "substation", "feeder", "r_ohm", "x_ohm", "km", "rk_ohm", "xk_ohm",
    "n_lines", "path_min_rating_mva", "feeder_sgen_mw", "feeder_load_mw",
    "dv_estimate_pu", "solar_over_rating", "feeder_gen_now_mw",
    "feeder_load_now_mw", "feeder_net_mw",
]
VN_KV = 20.0


def read_bus_features():
    return pd.read_csv(BUS_FEATURE_FILE, index_col="bus")


def add_features(scenarios, bus_features):
    """Model inputs for each row of `scenarios` (columns bus, solar_mw,
    load_level, gen_level), using the fixed facts about its bus."""
    x = scenarios[["bus", "solar_mw", "load_level", "gen_level"]].join(
        bus_features, on="bus")
    # Physics hints:
    # rough voltage rise from the project, in per unit: P x R / V^2
    x["dv_estimate_pu"] = x.solar_mw * x.rk_ohm / VN_KV ** 2
    # project size compared with the weakest line on its path
    x["solar_over_rating"] = x.solar_mw / x.path_min_rating_mva
    # existing generation and load actually on the feeder right now, and
    # the balance between them (positive: the feeder exports power)
    x["feeder_gen_now_mw"] = x.gen_level * x.feeder_sgen_mw
    x["feeder_load_now_mw"] = x.load_level * x.feeder_load_mw
    x["feeder_net_mw"] = x.feeder_gen_now_mw - x.feeder_load_now_mw
    return x[FEATURES]


def passes(predicted):
    """Pass/fail from predicted numbers, with the limits from decisions.md."""
    return ((predicted["max_voltage_pu"] <= MAX_VOLTAGE_PU)
            & (predicted["max_line_loading"] <= MAX_LOADING_PERCENT)
            & (predicted["max_trafo_loading"] <= MAX_LOADING_PERCENT))


def load_model():
    return joblib.load(MODEL_FILE)


def predict(models, x):
    """Predicted highest voltage, line loading and trafo loading for rows `x`."""
    return pd.DataFrame({t: models[t].predict(x) for t in TARGETS}, index=x.index)


def model_capacity(models, bus_features, cases, max_mw=20.0, step=0.1, batch=1000):
    """Model-predicted capacity for each row of `cases` (columns bus,
    load_level, gen_level): the largest size, in 0.1 MW steps, below the
    first size the model says fails. Same rule as find_capacity.
    Works through `cases` in batches to keep memory use small.
    """
    parts = [_model_capacity_batch(models, bus_features, cases.iloc[i:i + batch], max_mw, step)
             for i in range(0, len(cases), batch)]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def _model_capacity_batch(models, bus_features, cases, max_mw, step):
    sizes = np.round(np.arange(0, max_mw + step / 2, step), 1)
    grid = cases.reset_index(drop=True).loc[np.repeat(np.arange(len(cases)), len(sizes))]
    grid = grid.reset_index(drop=True)
    grid["solar_mw"] = np.tile(sizes, len(cases))
    ok = passes(predict(models, add_features(grid, bus_features))).to_numpy()
    ok = ok.reshape(len(cases), len(sizes))
    # Index of the first failing size per case (len(sizes) if none fail).
    first_fail = np.where(ok.all(axis=1), len(sizes), (~ok).argmax(axis=1))
    capacity = np.where(first_fail == 0, 0.0, sizes[np.maximum(first_fail - 1, 0)])
    return pd.Series(capacity, index=cases.index)
