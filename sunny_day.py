"""A day of sunshine on the grid, for the "Play a sunny day" replay in app_v2.py.

If results/sunlight_2024.csv exists (NASA POWER satellite sunlight, from
fetch_sunlight.py), the replay uses the sunniest real day of 2024 at the
grid's location. Otherwise it uses a cloudless late-May day.

Every hour is a full AC power flow, with the same conventions as
sunlight.py's year check, except for demand:
  - sun share s = the hour's sunlight / the year's brightest hour
    (on the cloudless day, s is 1 at 13:00);
  - the solar already on the grid (all 153 generators in this grid are
    solar) runs at 0.8 x s, so the brightest hour matches the worst case;
  - the new project runs at its size x s;
  - demand follows an illustrative sunny-weekend shape (below): low around
    midday, highest in the evening. It is not measured data.
"""
import copy
import math

import pandapower as pp
import pandas as pd

from engine import check, load_grid
from sunlight import SUNLIGHT_FILE

HOURS = list(range(5, 22))  # 05:00 to 21:00, local time
LOCAL_OFFSET_H = 2  # the grid is in Germany: summer time is UTC+2
EXISTING_SOLAR_TOP = 0.8

# Demand on a sunny weekend, as a share of peak load. Illustrative.
WEEKEND_LOAD = {5: 0.22, 6: 0.22, 7: 0.24, 8: 0.27, 9: 0.25, 10: 0.20, 11: 0.16,
                12: 0.14, 13: 0.13, 14: 0.14, 15: 0.17, 16: 0.22, 17: 0.30,
                18: 0.38, 19: 0.42, 20: 0.40, 21: 0.35}


def clear_sky_day():
    """Sun share per hour on a cloudless late-May day at about 48 degrees north:
    sunrise about 05:30, sunset about 20:30, full sun at 13:00."""
    out = {}
    for h in HOURS:
        x = (h - 5.5) / 15.0
        out[h] = round(max(0.0, math.sin(math.pi * x)) ** 1.15, 3)
    return out


def real_day(path=SUNLIGHT_FILE):
    """(date, sun share per local hour) for the sunniest day in the NASA
    file, or None if the file isn't there or holds no usable hours."""
    if not path.exists():
        return None
    table = pd.read_csv(path, parse_dates=["time_utc"]).dropna(subset=["irradiance_w_m2"])
    if table.empty or table.irradiance_w_m2.max() <= 0:
        return None
    brightest = table.irradiance_w_m2.max()
    local = table.time_utc + pd.Timedelta(hours=LOCAL_OFFSET_H)
    table = table.assign(date=local.dt.date, hour=local.dt.hour)
    best = table.groupby("date").irradiance_w_m2.sum().idxmax()
    day = table[table.date == best].set_index("hour").irradiance_w_m2
    sun = {h: round(max(float(day.get(h, 0.0)), 0.0) / float(brightest), 3) for h in HOURS}
    return best, sun


def day_profile():
    """(short label, long label, sun share per hour) for the replay."""
    real = real_day()
    if real:
        date, sun = real
        when = f"{date.day} {date:%B}"
        return (f"the sunniest day of 2024 ({when})",
                f"{when} 2024: real sunlight from NASA satellite data", sun)
    return "a sunny late-May day", "A sunny late-May day (cloudless sky)", clear_sky_day()


def play_day(bus, mw, sun):
    """One solved grid per hour: a list of (hour, sun share, net, check result)."""
    base = load_grid()
    out = []
    for h in HOURS:
        net = copy.deepcopy(base)
        net.load["scaling"] = WEEKEND_LOAD[h]
        net.sgen["scaling"] = EXISTING_SOLAR_TOP * sun[h]
        if mw > 0 and sun[h] > 0:
            pp.create_sgen(net, bus, p_mw=mw * sun[h], name="New project")
        out.append((h, sun[h], net, check(net)))
    return out
