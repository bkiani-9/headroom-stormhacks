"""Download real 2024 hourly sunlight for the grid from NASA POWER.

Run it once, with internet, from the project folder:

    py -3.14 fetch_sunlight.py

It asks NASA POWER (power.larc.nasa.gov, no key needed) for 2024 hourly
all-sky surface shortwave irradiance (ALLSKY_SFC_SW_DWN, satellite-based,
CERES SYN1deg) at the average position of the grid's buses, and saves it
to results/sunlight_2024.csv. The apps only read that file and never
download anything, so they still work offline.
"""
import io
import json
from pathlib import Path

import pandas as pd
import requests

from engine import load_grid

OUT_FILE = Path(__file__).parent / "results" / "sunlight_2024.csv"
URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
MISSING = -999


def grid_centre():
    """The average (latitude, longitude) of every bus in the grid."""
    points = [json.loads(geo)["coordinates"] for geo in load_grid().bus.geo]
    lon = sum(p[0] for p in points) / len(points)
    lat = sum(p[1] for p in points) / len(points)
    return round(lat, 4), round(lon, 4)


def main():
    lat, lon = grid_centre()
    print(f"Asking NASA POWER for 2024 hourly sunlight at {lat} N, {lon} E ...")
    response = requests.get(URL, timeout=120, params={
        "parameters": "ALLSKY_SFC_SW_DWN", "community": "RE",
        "latitude": lat, "longitude": lon,
        "start": "20240101", "end": "20241231",
        "format": "CSV", "time-standard": "UTC",
    })
    response.raise_for_status()

    # The CSV starts with a header block; the data begins after "-END HEADER-".
    text = response.text.split("-END HEADER-", 1)[1].strip()
    raw = pd.read_csv(io.StringIO(text))
    table = pd.DataFrame({
        "time_utc": pd.to_datetime(dict(year=raw.YEAR, month=raw.MO, day=raw.DY, hour=raw.HR)),
        # Energy per hour, so also the average power in W/m^2 over that hour.
        "irradiance_w_m2": raw.ALLSKY_SFC_SW_DWN.where(raw.ALLSKY_SFC_SW_DWN != MISSING),
    })
    OUT_FILE.parent.mkdir(exist_ok=True)
    table.to_csv(OUT_FILE, index=False)

    daylight = (table.irradiance_w_m2 > 0).sum()
    print(f"Saved {len(table):,} hours to {OUT_FILE}")
    print(f"  daylight hours: {daylight:,}; missing hours: {table.irradiance_w_m2.isna().sum()}")
    print(f"  brightest hour: {table.irradiance_w_m2.max():.0f} W/m2 on "
          f"{table.time_utc[table.irradiance_w_m2.idxmax()]:%d %b %H:00} UTC")


if __name__ == "__main__":
    main()
