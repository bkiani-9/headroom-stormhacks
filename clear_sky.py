"""The 3% rule again, for a 2024 with no clouds at all.

How much of the extra room the 3% rule finds comes from the clouds that
NASA's satellites saw? To answer it, this runs the same rule, the same AI
model and the same physics recheck as peak_shaving.py, and changes only the
sunlight: a cloudless sky instead of the satellite data.

The cloudless sky is the Haurwitz (1945) clear-sky model: sunlight on flat
ground from the sun's height alone, averaged over each hour of 2024 at the
grid's location. As with the satellite data, only each hour's share of the
year's brightest hour matters.

Writes results/peak_shaving_clear.csv (one row per bus) and
results/peak_shaving_clear_check.json (the physics recheck and the counts).
Run once: py -3.14 clear_sky.py (about 10 minutes, no internet needed).
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import ml
from capacity import MAX_SEARCH_MW
from capacity_data import read_capacity
from engine import load_grid
from peak_shaving import GEN_PEAK, LOAD_LEVEL, OUT_FILE as RULE_FILE, recheck, rule_size

ROOT = Path(__file__).parent
OUT_FILE = ROOT / "results" / "peak_shaving_clear.csv"
CHECK_FILE = ROOT / "results" / "peak_shaving_clear_check.json"
S_GRID = np.linspace(0, 1, 101)   # sun shares the AI model is asked about
SIZES_MW = (3, 5)                 # project sizes the summary counts


def grid_centre():
    """The average (latitude, longitude) of every bus, as in fetch_sunlight.py."""
    points = [json.loads(geo)["coordinates"] for geo in load_grid().bus.geo]
    return (sum(p[1] for p in points) / len(points), sum(p[0] for p in points) / len(points))


def clear_sky_sunlight(lat, lon, year=2024):
    """Daylight hours of a cloudless `year`: time_utc, irradiance_w_m2, sun_share.

    Each hour is the average of six points in it, so sunrise and sunset hours
    count partly, as in NASA's hourly data. The sun's position uses NOAA's
    standard approximations."""
    start = pd.Timestamp(f"{year}-01-01")
    hours = pd.date_range(start, f"{year}-12-31 23:00", freq="h")
    days = 366 if hours[-1].is_leap_year else 365
    minutes = np.arange(5, 60, 10)                       # 6 points per hour
    t = hours.values[:, None] + minutes[None, :].astype("timedelta64[m]")
    t = pd.DatetimeIndex(t.ravel())
    n = t.dayofyear.to_numpy()
    h = t.hour.to_numpy() + t.minute.to_numpy() / 60
    g = 2 * np.pi / days * (n - 1 + (h - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    hour_angle = np.radians((h * 60 + eqtime + 4 * lon) / 4 - 180)
    phi = np.radians(lat)
    cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    ghi = np.where(cos_z > 0, 1098 * cos_z * np.exp(-0.057 / np.maximum(cos_z, 1e-6)), 0.0)
    table = pd.DataFrame({"time_utc": hours, "irradiance_w_m2": ghi.reshape(len(hours), -1).mean(axis=1)})
    table = table[table.irradiance_w_m2 > 0].copy()
    table["sun_share"] = table.irradiance_w_m2 / table.irradiance_w_m2.max()
    return table.reset_index(drop=True)


def capacity_curves(models, bus_features, buses):
    """AI-model capacity of every bus at each sun share in S_GRID: array [bus, share].
    The worst-case demand is fixed, so a bus's capacity depends on the hour
    only through its sun share; each hour is then read off this curve."""
    cases = pd.DataFrame({
        "bus": np.repeat(buses, len(S_GRID)),
        "load_level": LOAD_LEVEL,
        "gen_level": np.tile(GEN_PEAK * S_GRID, len(buses)),
    })
    caps = ml.model_capacity_bisect(models, bus_features, cases, max_mw=MAX_SEARCH_MW)
    return caps.to_numpy().reshape(len(buses), len(S_GRID))


def main():
    warnings.filterwarnings("ignore")
    lat, lon = grid_centre()
    sky = clear_sky_sunlight(lat, lon)
    sun = sky.sun_share.to_numpy()
    models, bus_features = ml.load_model(), ml.read_bus_features()
    firm = read_capacity().capacity_mw
    buses = [b for b in bus_features.index if b in firm.index]
    print(f"Cloudless 2024 at {lat:.3f} N, {lon:.3f} E: {len(sun):,} daylight hours, "
          f"{sun.sum():,.0f} full-output hours")
    curves = capacity_curves(models, bus_features, buses)

    rows, caps_by_bus = [], {}
    for i, b in enumerate(buses):
        caps = np.interp(sun, S_GRID, curves[i])
        caps_by_bus[b] = caps
        f = float(firm[b])
        size, share, hours, lost, produced = rule_size(caps, sun, f)
        firm_mwh = f * float(sun.sum())
        rows.append({"bus": int(b), "firm_mw": f, "rule_mw": size, "model_rule_mw": size,
                     "curtailed_share": round(share, 5), "curtailed_hours": hours,
                     "firm_mwh": round(firm_mwh, 1), "rule_mwh": round(produced - lost, 1),
                     "gain_mwh": round(produced - lost - firm_mwh, 1),
                     "checked_by": "model" if size > f + 1e-9 else "firm capacity"})
    table = pd.DataFrame(rows).set_index("bus")

    start = time.perf_counter()
    summary = recheck(table, caps_by_bus, sun)
    real = pd.read_csv(RULE_FILE).set_index("bus").reindex(table.index)
    summary.update({
        "rule": "curtailment at most 3% of the yearly energy (section 11(2) EnWG)",
        "sky": "cloudless (Haurwitz 1945 clear-sky model), hourly averages, 2024",
        "daylight_hours": int(len(sun)),
        "full_output_hours": round(float(sun.sum()), 1),
        "load_level": LOAD_LEVEL,
        "fit": {str(mw): {"firm": int((table.firm_mw >= mw).sum()),
                          "cloudless": int((table.rule_mw >= mw).sum()),
                          "satellite": int((real.rule_mw >= mw).sum())}
                for mw in SIZES_MW},
        "mean_mw": {"firm": round(float(table.firm_mw.mean()), 2),
                    "cloudless": round(float(table.rule_mw.mean()), 2),
                    "satellite": round(float(real.rule_mw.mean()), 2)},
        "recheck_seconds": round(time.perf_counter() - start),
    })
    OUT_FILE.parent.mkdir(exist_ok=True)
    table.reset_index().to_csv(OUT_FILE, index=False)
    CHECK_FILE.write_text(json.dumps(summary, indent=2))
    print(f"Saved {OUT_FILE} and {CHECK_FILE}.")
    for mw, f in summary["fit"].items():
        print(f"  {mw} MW fits: firm {f['firm']}, cloudless 3% rule {f['cloudless']}, "
              f"satellite 3% rule {f['satellite']}")
    m = summary["mean_mw"]
    print(f"  mean MW: firm {m['firm']}, cloudless {m['cloudless']}, satellite {m['satellite']}")


if __name__ == "__main__":
    main()
