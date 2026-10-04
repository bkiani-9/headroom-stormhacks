"""Headroom, second interface: the whole grid as a live one-line diagram.

Start it with run_v2.cmd: dark theme, on port 8502, so app.py can still
run on 8501 as a fallback. Same engine, model and results files as app.py;
nothing here changes them.
"""
import base64
import json
import re
import struct
import time
import zlib

import numpy as np
import pandas as pd
import streamlit as st

import ml
import sunlight
from sunny_day import day_profile, play_day
from hyperbolic import disk_data, disk_layout, measure_fit, show_disk
from globe import show_globe
from motion import start_motion
from capacity_data import CAPACITY95_FILE, CAPACITY_FILE, read_capacity, read_capacity95
from control_room import (GREEN, ORANGE, RED, SUN, board_layout, capacity_board, explore_board,
                          flows_outward, path_lines, show_board, substation_of)
from peak_shaving import CHECK_FILE as RULE_CHECK_FILE, OUT_FILE as RULE_FILE
import clear_sky
from clear_sky import OUT_FILE as CLEAR_FILE
from capacity import MAX_SEARCH_MW
from engine import (MAX_LOADING_PERCENT, MAX_VOLTAGE_PU, add_project, check,
                    farthest_bus, load_grid, solve)

st.set_page_config(page_title="Headroom", layout="wide", initial_sidebar_state="collapsed")

# Grid conditions: (load level, level of the generators already on the grid).
CONDITIONS = {
    "Worst case": (0.10, 0.80),
    "Typical": (0.35, 0.40),
    "Best case": (0.60, 0.00),
}
CONDITION_PHRASES = {
    "Worst case": "In the worst case",
    "Typical": "On a typical day",
    "Best case": "In the best case",
}
CONDITION_NOTES = {
    "Worst case": "A bright, quiet day: demand at 10% of its peak and the solar already on "
                  "the grid at 80%. The least room for new solar.",
    "Typical": "An ordinary day: demand at 35% of its peak, the other solar at 40%.",
    "Best case": "A dark, busy day: demand at 60% of its peak and no other solar. The most "
                 "room for new solar.",
}
VIEWS = ["Try a project", "Capacity map", "From orbit", "Electrical space", "Proof"]
LAYERS = ["Firm and flexible", "Firm and the 3% rule"]  # the Capacity map's two layers
INTROS = {
    "Try a project": "Each box is a connection point on the grid, called a bus. Pick one, choose "
                     "the size of a new solar project, and see whether the grid can take it.",
    "Capacity map": "The same question for every connection point at once: can it take a project "
                    "of this size? Green means always, orange or yellow means with limits on its "
                    "output, red means no.",
    "From orbit": "The sunlight behind the 3% rule comes from NASA's satellites. This is what they "
                  "saw over the grid in 2024, and what the clouds change.",
    "Electrical space": "The same grid, drawn so that distance means electrical resistance, not "
                        "kilometres. The farther out a point sits, the less new solar it can take.",
    "Proof": "Can you trust the AI? Here it is tested against the physics engine on cases it "
             "never saw, and raced against it.",
}
HOW_IT_WORKS = """
1. **A real kind of grid.** A 20 kV distribution grid in Germany (pandapower's MV Oberrhein
   model): 177 connection points, 2 substations and 153 solar plants already connected.
2. **A physics engine.** For any new project, an AC power flow, the standard calculation grid
   operators use, works out every voltage and how full every line is. It's the reference
   answer, but each check takes a moment.
3. **An AI model for speed.** Trained on 60,000 physics checks, it answers almost instantly,
   so it can search the whole grid under thousands of conditions.
4. **Physics has the last word.** Every capacity on the maps was rechecked with the physics
   engine.

**Three ways to connect a project**
- **Firm:** it fits even at the worst moment of the year.
- **Flexible:** it fits in 95% of grid conditions, and is held back in the rest.
- **3% rule:** German grid operators may plan on curtailing up to 3% of a plant's yearly
  energy (§11(2) EnWG). Headroom works this out hour by hour with 2024 NASA satellite sunlight.

**Who it's for:** grid operators deciding where new solar can connect, and developers and
community energy groups planning projects.
"""
DEMO_MW = 3.0  # what the one-click button adds
MAP_SIZE_MW = 5.0  # starting project size on the Capacity map
RACE_CONDITIONS = 100  # grid conditions per bus in the speed test
SOLVES_PER_SEARCH = 10  # power flows in one find_capacity search: 2 edge checks + 8 halvings


# ---------------------------------------------------------------- data

@st.cache_resource
def base_grid():
    net = load_grid()
    solve(net)
    return net


@st.cache_resource
def grid_for(condition):
    """The grid under a named condition, solved, without any new project."""
    load, gen = CONDITIONS[condition]
    net = load_grid()
    net.load["scaling"], net.sgen["scaling"] = load, gen
    solve(net)
    return net


@st.cache_resource(max_entries=400)
def with_solar(bus, mw, condition):
    """The grid with `mw` of new solar at `bus`, solved and checked.
    Remembered, so dragging the slider back is instant."""
    net = add_project(grid_for(condition), bus, mw)
    start = time.perf_counter()
    result = check(net)  # runs the AC power flow
    result["ms"] = (time.perf_counter() - start) * 1000
    return net, result


@st.cache_resource(max_entries=30)
def day_replay(bus, mw, sunlight_file_time):
    """The "Play a sunny day" replay: labels, and one solved grid per hour."""
    short, long_label, sun = day_profile()
    return short, long_label, play_day(bus, mw, sun)


@st.cache_resource
def layout():
    return board_layout(base_grid())


@st.cache_resource
def disk():
    """The grid laid out in the hyperbolic plane (hyperbolic.py)."""
    return disk_layout(base_grid(), layout())


@st.cache_data
def hyperbolic_fit():
    """(error of the hyperbolic drawing, error of a flat drawing fitted to the same distances)."""
    return measure_fit(base_grid(), layout())


@st.cache_resource
def far_bus():
    return farthest_bus(base_grid())


@st.cache_data
def capacities(firm_changed, flex_changed, rule_changed, clear_changed):
    """Firm (worst case, results/capacity.csv), flexible (95% of
    conditions, results/capacity95.csv) and, if peak_shaving.py has run,
    3% rule (results/peak_shaving.csv) capacity per bus, in one table.
    If clear_sky.py has run, also the 3% rule in a cloudless year."""
    firm = read_capacity()
    flex = read_capacity95()
    table = pd.DataFrame({
        "firm": firm.capacity_mw,
        "firm_limit": firm.limit,
        "flex": np.maximum(flex.capacity_mw.reindex(firm.index).fillna(0), firm.capacity_mw),
        "pass_share": flex.pass_share.reindex(firm.index),
    })
    if RULE_FILE.exists():
        rule = pd.read_csv(RULE_FILE).set_index("bus").reindex(firm.index)
        table["rule"] = np.maximum(rule.rule_mw.fillna(0), firm.capacity_mw)
        table["rule_share"] = rule.curtailed_share.fillna(0)
        table["rule_hours"] = rule.curtailed_hours.fillna(0).astype(int)
        table["rule_gain"] = rule.gain_mwh.fillna(0)
        if CLEAR_FILE.exists():
            clear = pd.read_csv(CLEAR_FILE).set_index("bus").reindex(firm.index)
            table["clear"] = np.maximum(clear.rule_mw.fillna(0), firm.capacity_mw)
    return table


@st.cache_data
def rule_check(changed):
    """peak_shaving.py's power flow recheck (results/peak_shaving_check.json)."""
    return json.loads(RULE_CHECK_FILE.read_text()) if RULE_CHECK_FILE.exists() else None


@st.cache_resource
def model_and_features():
    return ml.load_model(), ml.read_bus_features()


def file_time(path):
    return path.stat().st_mtime if path.exists() else 0


# Kept on disk, so a bus and size checked once comes back instantly, even
# after the app restarts. The sunlight file's time is part of the key.
@st.cache_data(persist="disk", show_spinner=False)
def sunlight_year(bus, mw, sunlight_file_time):
    """A year of real 2024 sunlight at `bus` with `mw` of solar (sunlight.py)."""
    models, features = model_and_features()
    return sunlight.year_check(bus, mw, models, features)


@st.cache_data
def sky_grid(sunlight_file_time):
    """2024's satellite sunlight in W/m² as [hour of day, day of year], in local
    standard time (UTC+1, no daylight saving), so noon stays mid-chart all year."""
    table = pd.read_csv(sunlight.SUNLIGHT_FILE, parse_dates=["time_utc"])
    local = table.time_utc + pd.Timedelta(hours=1)
    keep = (local.dt.year == 2024).to_numpy()
    grid = np.zeros((24, 366))
    grid[local.dt.hour.to_numpy()[keep], local.dt.dayofyear.to_numpy()[keep] - 1] = (
        table.irradiance_w_m2.fillna(0).to_numpy()[keep])
    return grid


@st.cache_data
def cloudless_grid():
    """The same year with no clouds (clear_sky.py's Haurwitz model), laid out
    like sky_grid()."""
    lat, lon = clear_sky.grid_centre()
    table = clear_sky.clear_sky_sunlight(lat, lon)
    local = table.time_utc + pd.Timedelta(hours=1)
    keep = (local.dt.year == 2024).to_numpy()
    grid = np.zeros((24, 366))
    grid[local.dt.hour.to_numpy()[keep], local.dt.dayofyear.to_numpy()[keep] - 1] = (
        table.irradiance_w_m2.to_numpy()[keep])
    return grid


@st.cache_data
def grid_place():
    """The grid's centre, its 20 kV connection points and the lines between them,
    all as (latitude, longitude)."""
    net = load_grid()
    mv = net.bus.index[net.bus.vn_kv == 20]
    where = {int(b): tuple(json.loads(g)["coordinates"][::-1]) for b, g in net.bus.geo[mv].dropna().items()}
    lines = tuple((where[int(f)], where[int(t)]) for f, t in zip(net.line.from_bus, net.line.to_bus)
                  if int(f) in where and int(t) in where)
    lat, lon = clear_sky.grid_centre()
    return lat, lon, tuple(where.values()), lines


# ---------------------------------------------------------------- look

CSS = """
<style>
:root { --ink: #FFFFFF; --ink2: #E4E4E4; --ink3: #C2C2C2; --edge: #5C5C5C; --panel: #141414;
        --green: #63C132; --orange: #FFA61A; --red: #FF4D57; }
html, body, .stApp, [data-testid="stAppViewContainer"] {
  font-family: "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif; }
.stApp { background: #0A0A0A; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"] { padding: 1rem 1.5rem 1.4rem; max-width: 1720px; }
[data-testid="stVerticalBlock"] { gap: .8rem; }
[data-testid="stWidgetLabel"] p { font-size: .9rem; color: var(--ink3); }
.st-key-toggle button, .st-key-race button, .st-key-play_day button, .st-key-toggle_space button { border: 1.5px solid #A0A0A0; border-radius: 12px;
    background: transparent; padding: .55rem 1.4rem; min-height: 0; }
.st-key-toggle button p, .st-key-play_day button p { font-size: 1.35rem; color: #FFFFFF; }
.st-key-toggle_space button p { font-size: 1.1rem; color: #FFFFFF; }
.st-key-play_day button { border-color: #E8BE3A; }
.st-key-play_day button:hover { border-color: #FFD34D; background: #1A1A1A; }
.st-key-race button p { font-size: 1.1rem; color: #FFFFFF; }
.st-key-toggle button:hover, .st-key-race button:hover { border-color: #FFFFFF; background: #1A1A1A; }
.hr-head { display: flex; align-items: center; gap: 14px; }
.hr-head svg { flex: none; }
.hr-name { font-size: 1.85rem; font-weight: 600; line-height: 1; color: var(--ink); }
.hr-line { font-size: .98rem; color: var(--ink3); margin-top: 5px; }
.hr-story { font-size: 1.55rem; line-height: 1.3; color: #FFFFFF; max-width: 34em; margin-bottom: 1.2rem; }
.hr-where { color: var(--ink2); font-size: .93rem; line-height: 1.45; margin-top: -.35rem; }
.hr-where b { color: var(--ink); font-weight: 600; }
.hr-note { color: var(--ink3); font-size: .85rem; line-height: 1.4; margin-top: -.45rem; }
.hr-lamps { display: grid; grid-template-columns: 12px 1fr auto; column-gap: 12px; row-gap: 10px;
            align-items: center; padding: 13px 16px; border-radius: 12px; border: 1px solid var(--edge);
            background: var(--panel); font-size: .98rem; }
.hr-lamp { width: 11px; height: 11px; border-radius: 50%; background: currentColor;
           box-shadow: 0 0 8px currentColor; }
.hr-lamps .n { color: var(--ink); }
.hr-lamps .n span { color: var(--ink3); font-size: .85rem; margin-left: 6px; }
.hr-lamps .v { font-variant-numeric: tabular-nums; font-weight: 600; text-align: right; }
.hr-lamps .v span { color: var(--ink3); font-weight: 400; font-size: .85rem; margin-left: 6px; }
/* The capacity bar sits right under the slider, on exactly the same scale: the slider's track
   is inset 6px at each end, and its own min/max labels are replaced by the bar's. */
.st-key-mw [data-testid="stSliderTickBar"] { display: none; }
.hr-gauge { margin-top: -1.2rem; }
.hr-gauge .wrap { position: relative; margin: 0 6px; }
.hr-gauge .scale { display: flex; justify-content: space-between; margin: 6px 6px 0;
                   font-size: .8rem; color: var(--ink3); font-variant-numeric: tabular-nums; }
.hr-gauge .track { position: relative; height: 8px; border-radius: 4px; background: #2C2C2C; overflow: hidden; }
.hr-gauge .track i { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 4px; }
.hr-gauge .mark { position: absolute; top: -4px; width: 3px; height: 16px; border-radius: 2px;
                  background: #FFFFFF; transform: translateX(-1px); }
.hr-gauge .keys { display: flex; gap: 6px 14px; margin: 8px 6px 0; font-size: .84rem; color: var(--ink3); flex-wrap: wrap; }
.hr-gauge .keys i { display: inline-block; width: 9px; height: 9px; border-radius: 2.5px; margin-right: 6px; }
.hr-counts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.hr-count { padding: 13px 16px; border-radius: 12px; border: 1px solid var(--edge); background: var(--panel); }
.hr-count .big { font-size: 2.3rem; font-weight: 600; line-height: 1; font-variant-numeric: tabular-nums; }
.hr-count .big small { font-size: 1rem; color: var(--ink3); font-weight: 400; margin-left: 5px; }
.hr-count .lbl { color: var(--ink2); font-size: .95rem; line-height: 1.35; margin-top: 7px; }
.hr-count.firm { border-color: rgba(99,193,50,.8); }
.hr-count.firm .big { color: #7EDB4C; }
.hr-count.flex { border-color: rgba(255,166,26,.8); }
.hr-count.flex .big { color: var(--orange); }
.hr-count.rule { border-color: rgba(255,211,77,.8); }
.hr-count.rule .big { color: #FFD34D; }
.hr-tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.hr-tile { padding: 14px 16px; border-radius: 12px; border: 1px solid var(--edge); background: var(--panel); }
.hr-tile .big { font-size: 2.2rem; font-weight: 600; line-height: 1.05; font-variant-numeric: tabular-nums; color: var(--ink); }
.hr-tile .lbl { color: var(--ink2); font-size: .92rem; line-height: 1.35; margin-top: 6px; }
.hr-race { padding: 14px 16px; border-radius: 12px; border: 1px solid var(--edge); background: var(--panel); }
.hr-lane + .hr-lane { margin-top: 14px; }
.hr-lane .top { display: flex; justify-content: space-between; gap: 10px; font-size: .95rem; margin-bottom: 6px; }
.hr-lane .top b { font-weight: 600; color: var(--ink); }
.hr-lane .top span { color: var(--ink2); font-variant-numeric: tabular-nums; text-align: right; }
.hr-lane .bar { height: 12px; border-radius: 6px; background: #262626; overflow: hidden; }
.hr-lane .bar i { display: block; height: 100%; min-width: 3px; border-radius: 6px; }
.hr-lane.model .bar i { background: var(--green); }
.hr-lane.solver .bar i { background: var(--orange); }
.hr-race .sum { margin-top: 14px; font-size: .95rem; color: var(--ink2); line-height: 1.5; }
.hr-race .sum b { color: var(--ink); font-weight: 600; }
.hr-plot { border-radius: 12px; border: 1px solid var(--edge); background: #F4F3EF; padding: 10px; }
.hr-plot img { width: 100%; display: block; border-radius: 6px; }
.hr-copy { color: var(--ink2); font-size: .98rem; line-height: 1.55; max-width: 62ch; }
.hr-h { font-size: 1.2rem; font-weight: 600; color: var(--ink); margin-top: .6rem; }
.hr-warn { border: 1px solid rgba(242,163,36,.5); border-radius: 12px; padding: 10px 14px; color: #F5C978; font-size: .92rem; }
/* Streamlit pulls each HTML block up by 1rem; these blocks give that back so gaps stay even. */
.hr-after { margin: .2rem 0 1rem; }
.hr-after .hr-verdict + .hr-where { margin-top: .35rem; }
.hr-after .hr-where + .hr-note { margin-top: .3rem; }
.hr-after .hr-note + .hr-verdict { margin-top: .55rem; }
.hr-after > .hr-note:first-child { margin-top: 0; }
.hr-lamps, .hr-counts, .hr-tiles, .hr-copy, .hr-race { margin-bottom: 1rem; }
.st-key-proof_left, .st-key-space_text { max-width: 700px; }
.st-key-proof_left .hr-copy, .st-key-space_text .hr-copy { max-width: none; }
.hr-plot { width: fit-content; max-width: 100%; }
.hr-plot img { width: auto; max-width: 100%; max-height: max(420px, calc(100vh - 330px)); }
.hr-last { margin-bottom: 1rem; }
.st-key-controls { border-top: 1px solid #2A2A2A; padding-top: 1rem; margin-top: .3rem; }
.st-key-sun_check button { border: 1.5px solid #A0A0A0; border-radius: 12px; background: transparent;
                           padding: .4rem 1.1rem; min-height: 0; }
.st-key-sun_check button:hover { border-color: #FFFFFF; background: #1A1A1A; }
.hr-head { margin-bottom: .85rem; }
.hr-intro { color: var(--ink2); font-size: 1rem; line-height: 1.45; max-width: 80ch; padding-top: .35rem;
            margin-bottom: 1.2rem; }
.hr-verdict { color: var(--ink); font-size: 1.12rem; font-weight: 600; line-height: 1.35; max-width: 40em; }
[data-testid="stPopoverBody"] { max-width: 560px; }
.hr-sky { margin: .6rem 0 1rem; }
.hr-sky .plot { position: relative; padding-left: 2.9rem; }
.hr-sky .plot img { display: block; width: 100%; height: 132px; max-width: none; object-fit: fill;
                    image-rendering: pixelated; border-radius: 3px; }
.hr-sky .hours span { position: absolute; left: 0; transform: translateY(-50%); font-size: .74rem;
                      color: var(--ink3); }
.hr-sky .months { display: grid; margin: .3rem 0 0 2.9rem; font-size: .74rem; color: var(--ink3); }
.hr-sky .keys { display: flex; flex-wrap: wrap; gap: .35rem 1.3rem; margin: .55rem 0 0 2.9rem;
                font-size: .82rem; color: var(--ink2); }
.hr-sky .key { display: inline-flex; align-items: center; gap: .45rem; }
.hr-sky .key i { display: inline-block; width: 12px; height: 12px; border-radius: 2px; }
.hr-sky .key i.ramp { width: 64px; height: 10px; }
.hr-sky .hr-note { margin: .45rem 0 0 2.9rem; }
.hr-sky.orbit { margin: .35rem 0 1.1rem; }
.hr-sky.orbit .hr-note { max-width: 70ch; }
.hr-globe svg { display: block; width: 100%; height: auto; }
.hr-count.plain .big { color: var(--ink); }
.st-key-view button { padding-left: .55rem; padding-right: .55rem; }
.st-key-view button p { font-size: .9rem; white-space: nowrap; }
/* Opening: a branded veil while the first page is built, then the blocks rise in, one after
   another. motion.py says when the page is ready; without it, the veil lifts after 6 s. */
:root { --hr-ease: cubic-bezier(.16, 1, .3, 1); }
.hr-veil { position: fixed; inset: 0; z-index: 1000002; background: #0A0A0A; display: flex;
           align-items: center; justify-content: center; pointer-events: none;
           animation: hrVeilLate .7s ease 6s forwards; }
html.hr-ready .hr-veil { animation: hrVeilGo .9s ease .25s forwards; }
.hr-veil .mark { display: flex; align-items: center; gap: 24px;
                 animation: hrMarkIn 1s var(--hr-ease) .05s backwards; }
.hr-veil .mark svg { width: 96px; height: auto; }
.hr-veil .word { font-size: 2.7rem; font-weight: 600; color: #FFFFFF; line-height: 1; }
.hr-veil .sub { font-size: 1rem; color: #C8C8C8; margin-top: 12px; }
.hr-veil .load { height: 2px; width: 200px; margin-top: 18px; background: #262626; overflow: hidden;
                 border-radius: 2px; }
.hr-veil .load i { display: block; height: 100%; width: 38%; background: #63C132; border-radius: 2px;
                   animation: hrLoad 1.15s ease-in-out infinite; }
@keyframes hrVeilLate { to { opacity: 0; visibility: hidden; } }
@keyframes hrVeilGo { 0% { opacity: 1; } 100% { opacity: 0; visibility: hidden; transform: scale(1.04); } }
@keyframes hrMarkIn { from { opacity: 0; transform: translateY(16px) scale(.96); filter: blur(6px); } }
@keyframes hrLoad { from { transform: translateX(-100%); } to { transform: translateX(265%); } }
/* Blocks: hidden until the page is ready (4 s at most), then they rise in one after another.
   Each view sits in its own container, so switching views plays the same entrance. */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :not(:first-child):not(:has([class*="st-key-view_"])),
[class*="st-key-view_"] > * {
  animation: hrRiseLate .9s var(--hr-ease) 4s backwards; }
html.hr-ready [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :not(:first-child):not(:has([class*="st-key-view_"])) {
  animation: hrRise 1.15s var(--hr-ease) .35s backwards; }
html.hr-ready [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :nth-child(3):not(:has([class*="st-key-view_"])) { animation-delay: .5s; }
html.hr-ready [class*="st-key-view_"] > * { animation: hrRise 1.15s var(--hr-ease) .65s backwards; }
html.hr-ready [class*="st-key-view_"] > :nth-child(2) { animation-delay: .82s; }
html.hr-ready [class*="st-key-view_"] > :nth-child(3) { animation-delay: .99s; }
html.hr-ready [class*="st-key-view_"] > :nth-child(n+4) { animation-delay: 1.16s; }
html.hr-settled [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :not(:first-child):not(:has([class*="st-key-view_"])) {
  animation-duration: .8s; animation-delay: 0s; }
html.hr-settled [class*="st-key-view_"] > * { animation-duration: .95s; animation-delay: .02s; }
html.hr-settled [class*="st-key-view_"] > :nth-child(2) { animation-delay: .14s; }
html.hr-settled [class*="st-key-view_"] > :nth-child(3) { animation-delay: .26s; }
html.hr-settled [class*="st-key-view_"] > :nth-child(n+4) { animation-delay: .38s; }
@keyframes hrRise { from { opacity: 0; transform: translateY(46px) scale(.985); filter: blur(7px); } }
@keyframes hrRiseLate { from { opacity: 0; transform: translateY(46px) scale(.985); filter: blur(7px); } }
/* Sections and panels further down rise into view as you scroll to them. */
@supports (animation-timeline: view()) {
  [class*="st-key-orbit_s"], [class*="st-key-proof_s"], [class*="st-key-proof_g"], .st-key-controls,
  .hr-count:not([class*="st-key-orbit_s"] .hr-count), .hr-lamps, .hr-race {
    animation: hrReveal linear both; animation-timeline: view(); animation-range: entry 0% cover 34%; }
}
@keyframes hrReveal { from { opacity: 0; transform: translateY(90px) scale(.95); }
                      to { opacity: 1; transform: none; } }
/* The year charts on From orbit shimmer a little, like sunlight. */
.hr-sky.orbit .plot img { animation: hrFlicker 5.2s ease-in-out infinite; }
.hr-sky.orbit.clear .plot img { animation-duration: 6.6s; animation-delay: -2.4s; }
.hr-sky.orbit .plot::after { content: ""; position: absolute; top: 0; bottom: 0; left: 2.9rem; right: 0;
  pointer-events: none; border-radius: 3px; mix-blend-mode: screen; background-size: 240% 100%;
  background-image: linear-gradient(100deg, rgba(255,240,190,0) 38%, rgba(255,240,190,.18) 50%, rgba(255,240,190,0) 62%);
  animation: hrSheen 7.5s linear infinite; }
.hr-sky.orbit.clear .plot::after { animation-delay: -3.7s; }
@keyframes hrFlicker { 0%, 100% { filter: brightness(1); } 7% { filter: brightness(1.17) saturate(1.1); }
  9% { filter: brightness(.9); } 12% { filter: brightness(1.08); } 16% { filter: brightness(1); }
  55% { filter: brightness(1.04); } 59% { filter: brightness(1.15) saturate(1.08); } 61% { filter: brightness(.93); }
  64% { filter: brightness(1.05); } 70% { filter: brightness(1); } }
@keyframes hrSheen { from { background-position: 130% 0; } to { background-position: -130% 0; } }
.st-key-hr_motion { display: none !important; }
@media (prefers-reduced-motion: reduce) {
  .hr-veil { display: none; }
  [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *, [class*="st-key-view_"] > *,
  [class*="st-key-orbit_s"], [class*="st-key-proof_s"], [class*="st-key-proof_g"], .hr-count, .hr-lamps,
  .hr-race, .st-key-controls, .hr-sky.orbit .plot img, .hr-sky.orbit .plot::after { animation: none !important; }
}
/* Sections: one idea per screen, with room to breathe. */
[class*="st-key-orbit_s"], [class*="st-key-proof_g"] { padding: 2.2rem 0 1.6rem; border-top: 1px solid #262626; }
.st-key-orbit_s1 { padding-top: 0; border-top: 0; }
.hr-sky.orbit .plot img { height: 170px; }
.hr-graph { border-radius: 14px; border: 1px solid var(--edge); background: #0F0F0F; padding: 12px; }
.hr-graph img { display: block; width: 100%; height: auto; filter: invert(.94) contrast(1.12); border-radius: 6px; }
.hr-globe.cell svg { max-width: 640px; margin: 0 auto; }
</style>
"""

LOGO = """<svg width="44" height="34" viewBox="0 0 44 34" aria-hidden="true">
<line x1="12" y1="17" x2="39" y2="17" stroke="#63C132" stroke-width="2.4"/>
<line x1="27" y1="17" x2="27" y2="6" stroke="#63C132" stroke-width="2.4"/>
<circle cx="19.5" cy="17" r="1.6" fill="#fff"/><circle cx="27" cy="11" r="1.6" fill="#fff"/>
<rect x="1.5" y="9" width="13" height="16" rx="3.5" fill="#161616" stroke="#A6A6A6" stroke-width="1.4"/>
<rect x="22" y="12" width="10" height="10" rx="2.6" fill="#161616" stroke="#A6A6A6" stroke-width="1.4"/>
<rect x="22" y="1" width="10" height="10" rx="2.6" fill="#161616" stroke="#A6A6A6" stroke-width="1.4"/>
<rect x="33" y="12" width="10" height="10" rx="2.6" fill="#161616" stroke="#FF4D57" stroke-width="1.6"/>
<rect x="35" y="18.4" width="6" height="1.8" rx=".9" fill="#FF4D57"/>
</svg>"""


def clean(markup):
    return re.sub(r"\s*\n\s*", " ", markup).strip()


VEIL = clean(f"""<div class="hr-veil" aria-hidden="true"><div><div class="mark">{LOGO}
<div><div class="word">Headroom</div><div class="sub">Finding room for new solar on the grid</div>
<div class="load"><i></i></div></div></div></div></div>""")


def html(markup):
    """Show a piece of HTML. Line breaks and indentation become single
    spaces first, so the markdown renderer can't mistake indented HTML
    for code."""
    st.markdown(clean(markup), unsafe_allow_html=True)


# Sunlight colours, from night through cloud and haze to full sun.
SKY_STOPS = ((0.0, (24, 24, 24)), (0.25, (78, 55, 18)), (0.55, (178, 122, 20)),
             (0.8, (242, 194, 52)), (1.0, (255, 230, 140)))
MONTHS = (("Jan", 31), ("Feb", 29), ("Mar", 31), ("Apr", 30), ("May", 31), ("Jun", 30),
          ("Jul", 31), ("Aug", 31), ("Sep", 30), ("Oct", 31), ("Nov", 30), ("Dec", 31))


def png_uri(rgb):
    """An RGB array [rows, columns, 3] of uint8 as a PNG data URI, written
    with the standard library only."""
    rows, cols, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(rows))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", cols, rows, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


SKY_ALT = "Every hour of 2024 over the grid, coloured by the sunlight NASA's satellites measured"


def sky_uri(grid, peak, marked_utc=()):
    """A sunlight grid [hour, day] as a PNG data URI, the hours in `marked_utc` in red."""
    share = np.clip(grid / peak, 0, 1)
    stops = np.array([s for s, _ in SKY_STOPS])
    colours = np.array([c for _, c in SKY_STOPS], dtype=float)
    rgb = np.stack([np.interp(share, stops, colours[:, k]) for k in range(3)], axis=-1)
    red = tuple(int(RED[i:i + 2], 16) for i in (1, 3, 5))
    for t in marked_utc:
        local = pd.Timestamp(t) + pd.Timedelta(hours=1)
        if local.year == 2024:
            rgb[local.hour, local.dayofyear - 1] = red
    return png_uri(np.round(rgb).astype(np.uint8))


def sky_axes(uri, height=132, alt=SKY_ALT):
    """The chart image with its hour labels on the left and months underneath."""
    months = "".join(f"<span>{m}</span>" for m, _ in MONTHS)
    widths = " ".join(f"{d}fr" for _, d in MONTHS)
    hours = "".join(f'<span style="top:{(h + .5) / 24:.2%}">{label}</span>'
                    for h, label in ((6, "6 am"), (12, "noon"), (18, "6 pm")))
    return (f'<div class="plot"><div class="hours">{hours}</div><img src="{uri}" alt="{alt}" '
            f'style="height:{height}px"></div>'
            f'<div class="months" style="grid-template-columns:{widths}">{months}</div>')


def sky_keys(peak, mark_label=""):
    """The colour key: the sunlight ramp, and the red hours if any."""
    ramp = ", ".join(f"rgb{c} {s:.0%}" for s, c in SKY_STOPS)
    mark = (f'<span class="key"><i style="background:{RED}"></i>{mark_label}</span>'
            if mark_label else "")
    return (f'<div class="keys"><span class="key"><i class="ramp" style="background:'
            f'linear-gradient(90deg, {ramp})"></i>sunlight, 0 to {peak:,.0f} W/m²</span>{mark}</div>')


def sky_chart(marked_utc=(), mark_label="", caption=""):
    """Every hour of 2024 over the grid, coloured by NASA's satellite sunlight:
    days across, hours down. The hours in `marked_utc` show in red."""
    grid = sky_grid(file_time(sunlight.SUNLIGHT_FILE))
    peak = float(grid.max())
    return (f'<div class="hr-sky">{sky_axes(sky_uri(grid, peak, marked_utc))}'
            f'{sky_keys(peak, mark_label)}<div class="hr-note">{caption}</div></div>')


@st.cache_data
def cell_svg(grid_lat, grid_lon, bus_points, lines):
    """One NASA data cell (1 degree square, to scale) with the grid inside it, and a
    close-up of the grid: its connection points and the lines between them."""
    km_x = 111.32 * np.cos(np.radians(grid_lat))   # km per degree of longitude here
    km_y = 110.57                                  # km per degree of latitude
    pts = np.array(bus_points)
    xs, ys = (pts[:, 1] - grid_lon) * km_x, (pts[:, 0] - grid_lat) * km_y   # km east, north

    def km(lat, lon):
        return (lon - grid_lon) * km_x, (lat - grid_lat) * km_y

    # The cell: 1 degree tall, drawn 290 px tall, centred on the grid.
    s1 = 290 / km_y
    cx, cy = 130, 185
    hw, hh = 0.5 * km_x * s1, 0.5 * km_y * s1
    small = "".join(f'<line x1="{cx + km(a[0], a[1])[0] * s1:.1f}" y1="{cy - km(a[0], a[1])[1] * s1:.1f}" '
                    f'x2="{cx + km(b[0], b[1])[0] * s1:.1f}" y2="{cy - km(b[0], b[1])[1] * s1:.1f}"/>'
                    for a, b in lines)
    fx0, fx1 = cx + xs.min() * s1 - 4, cx + xs.max() * s1 + 4
    fy0, fy1 = cy - ys.max() * s1 - 4, cy - ys.min() * s1 + 4
    # The close-up: the grid's extent fills a 290 px square.
    zx, zy, zs = 330, 40, 290
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.08
    s2 = zs / span
    mx, my = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2

    def zoom(lat, lon):
        x, y = km(lat, lon)
        return zx + zs / 2 + (x - mx) * s2, zy + zs / 2 - (y - my) * s2

    big = "".join('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}"/>'.format(*zoom(*a), *zoom(*b))
                  for a, b in lines)
    dots = "".join('<circle cx="{:.1f}" cy="{:.1f}" r="2.3"/>'.format(*zoom(la, lo)) for la, lo in bus_points)
    bar1, bar2 = 20 * s1, 5 * s2
    return f"""<svg viewBox="0 0 640 380" role="img"
aria-label="One NASA data cell, 1 degree by 1 degree, with the whole grid inside it, and a close-up of the grid's {len(bus_points)} connection points and lines">
<text x="{cx - hw:.1f}" y="22" fill="#F2F2F2" font-size="15" font-weight="600">One NASA data cell</text>
<rect x="{cx - hw:.1f}" y="{cy - hh:.1f}" width="{2 * hw:.1f}" height="{2 * hh:.1f}" rx="2"
fill="rgba(255,211,77,.08)" stroke="{SUN}" stroke-width="1.5"/>
<text x="{cx:.1f}" y="{cy + hh - 12:.1f}" fill="#D6D6D6" font-size="13" text-anchor="middle">1° × 1°: {km_y:.0f} × {km_x:.0f} km</text>
<g stroke="#7EDB4C" stroke-width=".8">{small}</g>
<rect x="{fx0:.1f}" y="{fy0:.1f}" width="{fx1 - fx0:.1f}" height="{fy1 - fy0:.1f}" fill="none" stroke="#F2F2F2"
stroke-width=".8" stroke-dasharray="2 2"/>
<line x1="{fx1:.1f}" y1="{fy0:.1f}" x2="{zx}" y2="{zy}" stroke="#8C8C8C" stroke-width=".8" stroke-dasharray="3 3"/>
<line x1="{fx1:.1f}" y1="{fy1:.1f}" x2="{zx}" y2="{zy + zs}" stroke="#8C8C8C" stroke-width=".8" stroke-dasharray="3 3"/>
<rect x="{zx}" y="{zy}" width="{zs}" height="{zs}" rx="10" fill="#0F1410" stroke="#5C5C5C"/>
<text x="{zx}" y="22" fill="#F2F2F2" font-size="15" font-weight="600">The grid: {len(bus_points)} connection points</text>
<g stroke="#7EDB4C" stroke-width="1.4" stroke-linecap="round">{big}</g>
<g fill="#EAF7E2">{dots}</g>
<line x1="{cx - hw:.1f}" y1="{cy + hh + 22:.1f}" x2="{cx - hw + bar1:.1f}" y2="{cy + hh + 22:.1f}" stroke="#D6D6D6" stroke-width="1.5"/>
<text x="{cx - hw + bar1 + 8:.1f}" y="{cy + hh + 26:.1f}" fill="#D6D6D6" font-size="13">20 km</text>
<line x1="{zx}" y1="{zy + zs + 22}" x2="{zx + bar2:.1f}" y2="{zy + zs + 22}" stroke="#D6D6D6" stroke-width="1.5"/>
<text x="{zx + bar2 + 8:.1f}" y="{zy + zs + 26}" fill="#D6D6D6" font-size="13">5 km</text>
</svg>"""


def lamp_colour(value, warn_from, limit):
    if value is None or value > limit + 1e-9:
        return RED
    if value > warn_from:
        return ORANGE
    return GREEN


# ---------------------------------------------------------------- page

st.markdown(CSS + VEIL, unsafe_allow_html=True)

net0 = base_grid()
board = layout()
mv_buses = [int(b) for b in net0.bus.index[net0.bus.vn_kv == 20]]
farthest = far_bus()
where = substation_of(board)
caps = capacities(file_time(CAPACITY_FILE), file_time(CAPACITY95_FILE), file_time(RULE_FILE),
                  file_time(CLEAR_FILE))
has_rule = "rule" in caps.columns
has_clear = "clear" in caps.columns

# Values that live across views. Streamlit forgets a widget's value on any
# run where the widget isn't drawn, so copy them over every run.
DEFAULTS = {"view": VIEWS[0], "bus": farthest, "mw": 0.0,
            "condition": "Typical", "cap_size": MAP_SIZE_MW, "cap_layer": LAYERS[0]}
for key, value in DEFAULTS.items():
    current = st.session_state.get(key)
    if current is None:  # first run, or a control that was cleared
        current = st.session_state.get(f"last_{key}", value)
    st.session_state[key] = current
    st.session_state[f"last_{key}"] = current


def bus_name(bus):
    if bus == farthest:
        return f"Bus {bus}, farthest from a substation"
    return f"Bus {bus}"


def pick_bus():
    st.selectbox("Connection point (bus)", mv_buses, key="bus", format_func=bus_name)


def where_line(bus):
    sub, hops = where.get(bus, ("its substation", 0))
    if hops == 0:
        place = f"The 20 kV side of {sub}."
    else:
        place = f"{hops} line{'s' if hops != 1 else ''} out from {sub}."
    if bus in caps.index:
        row = caps.loc[bus]
        return (f"{place} Takes <b>{row.firm:.1f} MW firm</b>, "
                f"<b>{row.flex:.1f} MW flexible</b>.")
    return place


def plain_limit(limit):
    """'voltage 1.0404 pu is above 1.04' -> 'voltage', and so on."""
    limit = (limit or "").strip()
    if not limit:
        return "nothing up to 20 MW"
    if limit.startswith("voltage"):
        return "voltage"
    if limit.startswith("Line"):
        return limit.split(" is ")[0].lower() + " overloading"
    return "the substation transformer overloading"


RULE_NOTE = ("The 3% rule: German grid operators may plan on curtailing up to 3% of a solar "
             "plant's yearly energy (§11(2) EnWG).")


def rule_size_text(row):
    """'3.1 MW', or '20.0 MW or more' at the top of the search range."""
    top = row.rule >= MAX_SEARCH_MW - 1e-9
    return f"{row.rule:.1f} MW" + (" or more" if top else "")


def rule_line(bus):
    """Try a project's line: firm capacity, then the 3% rule's."""
    row = caps.loc[bus]
    firm = f"Firm: <b>{row.firm:.1f} MW</b>."
    if row.firm >= MAX_SEARCH_MW - 1e-9:
        return f"{firm} That's the top of the search range, so the 3% rule can't add more."
    if row.rule <= row.firm + 1e-9:
        return f"{firm} The 3% rule adds nothing here."
    line = (f"{firm} Under the 3% rule: <b>{rule_size_text(row)}</b>, losing "
            f"<b>{row.rule_share:.1%}</b> of its energy.")
    if has_clear and row.rule > row.clear + 0.05:
        line += f" Without the clouds NASA's satellites saw, only <b>{row.clear:.1f} MW</b>."
    return line


def connection_verdict(bus, mw):
    """Which kinds of connection fit `mw` at `bus`, in one sentence."""
    if mw <= 0:
        return "Choose a project size to see which kinds of connection fit here."
    row = caps.loc[bus]
    size = f"{mw:g} MW"
    if mw <= row.firm + 1e-9:
        return f"{size} fits here as a firm connection, with no limits on its output."
    flex = mw <= row.flex + 1e-9
    rule = has_rule and mw <= row.rule + 1e-9
    if flex and rule:
        return (f"{size} fits here with a flexible connection or under Germany's 3% rule, "
                "but not as a firm connection.")
    if flex:
        return f"{size} fits here only with a flexible connection."
    if rule:
        return f"{size} fits here only under Germany's 3% rule."
    return (f"{size} is too big for this point under every option. Try a smaller project, or a "
            "point closer to a substation.")


def voltage_words(vm):
    """1.0487 -> '4.9% above normal'."""
    rise = (vm - 1.0) * 100
    return f"{abs(rise):.1f}% {'above' if rise >= 0 else 'below'} normal"


def rule_detail(row, daylight_hours):
    """The Capacity map's line about the 3% rule at one bus."""
    if row.rule <= row.firm + 1e-9:
        return "The 3% rule adds nothing here."
    hours = f" of {daylight_hours:,} daylight hours" if daylight_hours else " hours"
    return (f"Under the 3% rule <b>{rule_size_text(row)}</b>, curtailing {row.rule_share:.1%} of its "
            f"energy in {row.rule_hours:,}{hours}: <b>+{row.rule_gain:,.0f} MWh</b> of clean energy "
            f"a year.")


def toggle_solar():
    """The one-click button: add the demo project, or take it away."""
    st.session_state.mw = 0.0 if st.session_state.mw > 0 else DEMO_MW


def start_day():
    """The "Play a sunny day" button: play the replay on the next run."""
    st.session_state.play_id = st.session_state.get("play_id", 0) + 1
    st.session_state.play_now = True


def broken_limit(result):
    """The limit a failing check broke, in a few words."""
    if result["max_voltage"] is None:
        return "No stable state"
    if result["max_voltage"] > MAX_VOLTAGE_PU:
        return "Voltage past its limit"
    if result["max_line"] > MAX_LOADING_PERCENT:
        return "A line is overloaded"
    return "A transformer is overloaded"


def day_frames(hours, bus, mw):
    """Board frames for the replay: only what changes from hour to hour."""
    frames = []
    for h, sun, net, result in hours:
        if result["max_voltage"] is None:
            continue  # no solved grid to draw for this hour
        data = explore_board(net, board, bus, mw * sun)
        frames.append({
            "clock": f"{h:02d}:00", "sun": sun, "alarm": not result["passes"],
            "status": "Within every limit" if result["passes"] else broken_limit(result),
            "buses": [{k: b[k] for k in ("bar", "colour", "state")} for b in data["buses"]],
            "lines": [{k: x[k] for k in ("dir", "mw", "stroke")} for x in data["lines"]],
            "subs": [{k: x[k] for k in ("note", "over")} for x in data["subs"]],
        })
    return frames


def hour_span(hours):
    """[12, 13, 14] -> 'from 12:00 to 14:00'."""
    if len(hours) == 1:
        return f"at {hours[0]:02d}:00"
    if hours == list(range(hours[0], hours[-1] + 1)):
        return f"from {hours[0]:02d}:00 to {hours[-1]:02d}:00"
    return "at " + ", ".join(f"{h:02d}:00" for h in hours)


def day_story(short, hours, bus, mw):
    """One plain sentence about the replayed day."""
    failing = [h for h, s, net, r in hours if not r["passes"]]
    if mw > 0 and failing:
        results = [r for h, s, net, r in hours if not r["passes"]]
        if any(r["max_voltage"] is not None and r["max_voltage"] > MAX_VOLTAGE_PU for r in results):
            what = "pushes the voltage past its safe limit"
        else:
            what = "overloads part of the grid"
        n = len(failing)
        return (f"On {short}, {mw:g} MW at bus {bus} {what} {hour_span(failing)}. "
                f"A flexible connection would hold it back for just those {n} hour"
                f"{'s' if n != 1 else ''} and let it run freely the rest of the day.")
    if mw > 0:
        return (f"On {short}, {mw:g} MW at bus {bus} stays inside every limit all day, "
                "even at the midday peak.")
    if failing:
        return f"On {short}, the grid breaks a limit {hour_span(failing)} even without new solar."
    back = [h for h, s, net, r in hours
            if r["max_voltage"] is not None and (net.res_trafo.p_hv_mw < 0).any()]
    if back:
        return (f"On {short}, the solar already on this grid sends power back up through "
                f"the substations {hour_span(back)}, and every limit holds.")
    return f"On {short}, every limit holds all day."


def story(condition, bus, mw, before_net, after_net, result):
    """One plain sentence about what the power flow shows."""
    phrase = CONDITION_PHRASES[condition]
    path = path_lines(board, bus)
    if not path:
        return f"{phrase}: bus {bus} is the substation's own 20 kV busbar."
    before = flows_outward(before_net, path)
    if mw == 0:
        if all(before):
            return f"{phrase}, power flows out from the substation to bus {bus}."
        if not any(before):
            return (f"{phrase}, the other generators already push power back toward "
                    f"the substation from bus {bus}.")
        return (f"{phrase}, power flows out from the substation and meets power coming "
                f"back from the generators near bus {bus}.")
    if result["max_voltage"] is None:
        return (f"{mw:g} MW at bus {bus}: the physics finds no stable state. "
                "The grid can't carry it.")
    after = flows_outward(after_net, path)
    turned = sum(b and not a for b, a in zip(before, after))
    if turned == len(path):
        flow = "power now flows back toward the substation"
    elif turned:
        flow = f"power now flows back toward the substation over the last {turned} lines"
    elif not any(after):
        flow = "even more power flows back toward the substation"
    else:
        flow = "power still flows out to it"
    if result["passes"]:
        return f"{mw:g} MW at bus {bus}: {flow}, and every limit holds."
    broken = []
    if result["max_voltage"] > MAX_VOLTAGE_PU:
        broken.append("voltage rises past its safe limit")
    if result["max_line"] > MAX_LOADING_PERCENT:
        broken.append(f"line {result['worst_line']} overloads")
    if result["max_trafo"] > MAX_LOADING_PERCENT:
        broken.append("a substation transformer overloads")
    return f"{mw:g} MW at bus {bus}: {flow}, and {' and '.join(broken)}."


# Header: name, one line on what it does, and the view switch.
left, right = st.columns([1.15, 1], vertical_alignment="center")
with left:
    html(f"""<div class="hr-head">{LOGO}<div>
<div class="hr-name">Headroom</div>
<div class="hr-line">Where new solar can connect to the grid, worked out in seconds and checked against real physics</div>
</div></div>""")
with right:
    st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed",
                         required=True, width="stretch")

if st.get_option("theme.base") != "dark":
    html('<div class="hr-warn">Start this version with <b>run_v2.cmd</b> to get its dark '
         'theme: the light theme from config.toml makes the controls hard to read here.</div>')


def try_a_project():
    bus = int(st.session_state.bus)
    mw = round(float(st.session_state.mw), 1)
    condition = st.session_state.condition
    before_net, _ = with_solar(bus, 0.0, condition)
    net, result = with_solar(bus, mw, condition)
    solved = result["max_voltage"] is not None
    shown = net if solved else before_net

    data = explore_board(shown, board, bus, mw if solved else 0,
                         result["ms"] if solved else None)
    day_text = None
    if st.session_state.pop("play_now", False):
        with st.spinner("Solving the grid for every hour of the day..."):
            short, long_label, hours = day_replay(bus, mw, file_time(sunlight.SUNLIGHT_FILE))
            data["frames"] = day_frames(hours, bus, mw)
        data["play"] = st.session_state.play_id
        data["day_label"] = long_label
        day_text = day_story(short, hours, bus, mw)
    clicked = show_board(data)
    if clicked is not None and clicked != bus and clicked in mv_buses:
        st.session_state.bus = clicked
        st.rerun()

    story_col, lamp_col = st.columns([1.45, 1], gap="large")
    with story_col:
        if day_text:
            html(f'<div class="hr-story">{day_text}</div>')
        else:
            html(f'<div class="hr-story">{story(condition, bus, mw, before_net, net, result)}</div>')
        label = f"Add {DEMO_MW:g} MW of solar at bus {bus}" if mw == 0 else "Remove the solar"
        with st.container(horizontal=True, gap="small"):
            st.button(label, key="toggle", on_click=toggle_solar)
            st.button("Play a sunny day", key="play_day", on_click=start_day)
        # Everything under the buttons in one block, so the spacing stays even.
        parts = []
        if day_text:
            parts.append('<div class="hr-note">Every hour is a full physics check. Demand follows '
                         'an illustrative sunny-weekend pattern, low at midday and highest in the '
                         'evening.</div>')
        if bus in caps.index:
            parts.append(f'<div class="hr-verdict">{connection_verdict(bus, mw)}</div>')
            if has_rule:
                parts.append(f'<div class="hr-where">{rule_line(bus)}</div>'
                             f'<div class="hr-note">{RULE_NOTE} Worked out hour by hour over 2024 '
                             'with NASA satellite sunlight, holding demand at its minimum.</div>')
        if parts:
            html(f'<div class="hr-after">{"".join(parts)}</div>')
    with lamp_col:
        if solved:
            vmax_bus = int(net.res_bus.vm_pu.idxmax())
            rows = [
                ("Voltage", f"highest, at bus {vmax_bus}", result["max_voltage"],
                 voltage_words(result["max_voltage"]),
                 f"limit {(MAX_VOLTAGE_PU - 1) * 100:.0f}% above",
                 MAX_VOLTAGE_PU - 0.01, MAX_VOLTAGE_PU),
                ("Busiest power line", f"line {result['worst_line']}", result["max_line"],
                 f"{result['max_line']:.0f}% full", f"limit {MAX_LOADING_PERCENT:.0f}%",
                 70.0, MAX_LOADING_PERCENT),
                ("Substation transformer", "the busier one", result["max_trafo"],
                 f"{result['max_trafo']:.0f}% full", f"limit {MAX_LOADING_PERCENT:.0f}%",
                 70.0, MAX_LOADING_PERCENT),
            ]
            cells = "".join(
                f'<i class="hr-lamp" style="color:{lamp_colour(v, w, lim)}"></i>'
                f'<div class="n">{name}<span>{sub}</span></div>'
                f'<div class="v">{text}<span>{limit}</span></div>'
                for name, sub, v, text, limit, w, lim in rows)
            html(f'<div class="hr-lamps">{cells}</div>')

        # A year of real satellite sunlight for this bus and size.
        if st.button("Check a year of real sunlight", key="sun_check"):
            st.session_state.sun_request = (bus, mw)
        if st.session_state.get("sun_request") == (bus, mw):
            if not sunlight.SUNLIGHT_FILE.exists():
                html('<div class="hr-note hr-last">No sunlight data yet: run '
                     '<code>py -3.14 fetch_sunlight.py</code> once, with internet.</div>')
            elif mw == 0:
                html('<div class="hr-note hr-last">Add some solar first.</div>')
            else:
                with st.spinner("Checking every daylight hour of 2024..."):
                    r = sunlight_year(bus, mw, file_time(sunlight.SUNLIGHT_FILE))
                n = r["solver_fail_hours"]
                caption = ("Each column is a day of 2024 and each row an hour, coloured by the "
                           "sunlight NASA's satellites measured over this grid. Dark streaks are "
                           "cloudy days. "
                           + ("Red marks the hours this project would need curtailing." if n else
                              "No hour of 2024 would need curtailing at this size."))
                html(f'<div class="hr-where">With real 2024 sunlight (NASA satellite '
                     f'data), <b>{mw:g} MW</b> at bus {bus} would need curtailing in '
                     f'<b>{n:,}</b> of {r["daylight_hours"]:,} daylight '
                     f'hours, if demand stayed as low as the worst case.</div>'
                     + sky_chart(r["fail_times_utc"], "needs curtailing" if n else "", caption))

    controls = st.container(key="controls")
    c1, c2, c3 = controls.columns([1, 1.2, 1.2], gap="large")
    with c1:
        pick_bus()
        html(f'<div class="hr-where">{where_line(bus)}</div>')
    with c2:
        st.slider("Size of the new solar project", 0.0, 20.0, step=0.1, key="mw",
                  format="%.1f MW")
        if bus in caps.index:
            row = caps.loc[bus]
            scale = 20.0
            # Longest bar first, so each shorter one sits on top of it.
            layers = [(row.flex, ORANGE)] + ([(row.rule, SUN)] if has_rule else [])
            layers.sort(key=lambda layer: -layer[0])
            bars = "".join(f'<i style="width:{min(v / scale, 1) * 100:.1f}%;background:{c}"></i>'
                           for v, c in layers)
            rule_key = (f'<span><i style="background:{SUN}"></i>3% rule up to {row.rule:.1f} MW</span>'
                        if has_rule else "")
            html(f"""<div class="hr-gauge"><div class="wrap"><div class="track">{bars}
<i style="width:{min(row.firm / scale, 1) * 100:.1f}%;background:{GREEN}"></i>
</div><div class="mark" style="left:{min(mw / scale, 1) * 100:.1f}%"></div></div>
<div class="scale"><span>0 MW</span><span>{scale:g} MW</span></div>
<div class="keys"><span><i style="background:{GREEN}"></i>Firm up to {row.firm:.1f} MW</span>
<span><i style="background:{ORANGE}"></i>Flexible up to {row.flex:.1f} MW</span>{rule_key}</div></div>""")
    with c3:
        st.segmented_control("Grid condition", list(CONDITIONS), key="condition",
                             required=True, width="stretch")
        html(f'<div class="hr-note">{CONDITION_NOTES[condition]}</div>')


def capacity_map():
    bus = int(st.session_state.bus)
    size = float(st.session_state.cap_size)
    rule = has_rule and st.session_state.cap_layer == LAYERS[1]
    second = caps.rule if rule else caps.flex
    check = rule_check(file_time(RULE_CHECK_FILE)) if rule else None

    clicked = show_board(capacity_board(base_grid(), board, size, caps.firm.to_dict(),
                                        second.to_dict(), bus, kind="rule" if rule else "flex",
                                        gains=caps.rule_gain.to_dict() if rule else None))
    if clicked is not None and clicked != bus and clicked in mv_buses:
        st.session_state.bus = clicked
        st.rerun()

    firm_n = int((caps.firm >= size - 1e-9).sum())
    flex_n = int((caps.flex >= size - 1e-9).sum())
    second_n = int((second >= size - 1e-9).sum())
    total = len(caps)
    controls, counts = st.columns([1, 2.1], gap="large")
    with controls:
        if has_rule:
            st.segmented_control("Layer", LAYERS, key="cap_layer", required=True, width="stretch")
        st.slider("Project size at every connection point", 0.5, 20.0, step=0.5,
                  key="cap_size", format="%.1f MW")
        pick_bus()
        if bus in caps.index:
            row = caps.loc[bus]
            if rule:
                hours = check.get("daylight_hours") if check else None
                html(f"""<div class="hr-where">Firm <b>{row.firm:.1f} MW</b>, limited by
{plain_limit(row.firm_limit)}. {rule_detail(row, hours)}</div>""")
            else:
                share = ("" if pd.isna(row.pass_share)
                         else f", passing {row.pass_share:.0%} of sampled conditions")
                html(f"""<div class="hr-where">Firm <b>{row.firm:.1f} MW</b>, limited by
{plain_limit(row.firm_limit)}. Flexible <b>{row.flex:.1f} MW</b>{share}.</div>""")
    with counts:
        if rule:
            html(f"""<div class="hr-counts">
<div class="hr-count firm"><div class="big">{firm_n}<small>of {total}</small></div>
<div class="lbl">connection points take {size:g} MW firm: it fits even at the worst moment</div></div>
<div class="hr-count rule"><div class="big">{second_n}<small>of {total}</small></div>
<div class="lbl">take it under the 3% rule, losing at most 3% of its yearly energy</div></div>
<div class="hr-count"><div class="big">+{second_n - firm_n}</div>
<div class="lbl">more points open up under the 3% rule, with no new wires</div></div></div>""")
            sky = ""
            if sunlight.SUNLIGHT_FILE.exists():
                cloud = ""
                if has_clear:
                    clear_n = int((caps.clear >= size - 1e-9).sum())
                    if second_n > clear_n:
                        cloud = (f'<div class="hr-where" style="margin:.1rem 0 0"><b>The clouds '
                                 f'matter.</b> In a cloudless year, the same rule would open only '
                                 f'<b>+{clear_n - firm_n}</b> points at {size:g} MW, not '
                                 f'+{second_n - firm_n}. Curtailment only bites in full sun, and '
                                 f'the satellite data shows how rare full sun is.</div>')
                sky = cloud + sky_chart(caption=(
                    "The sunlight behind the 3% rule: every hour of 2024 over this grid, as NASA's "
                    "satellites measured it (CERES data, via NASA POWER). Days run left to right, "
                    "hours top to bottom, and dark streaks are cloudy days."))
            recheck = ""
            if check:
                recheck = (f" Rechecked with the physics engine at {check['buses_checked']} points, "
                           f"{check['hours_checked']:,} hours: {check['buses_kept']} kept the AI's "
                           f"size, {check['buses_lowered']} came down by at most "
                           f"{check['most_lowered_mw']:g} MW.")
            cloudless = (" The cloudless year runs the same rule, AI model and physics recheck on a "
                         "clear-sky model (Haurwitz) instead of the satellite data." if has_clear else "")
            html(f"""{sky}<div class="hr-note" style="margin-top:.2rem">{RULE_NOTE} For every daylight
hour of 2024, from NASA satellite sunlight, the AI model finds how much each point can take, and
anything above that is curtailed. Assumptions: demand held at its minimum (10% of peak) all year,
the solar already on the grid follows the same sunlight and is never curtailed, hourly
data.{recheck}{cloudless}</div>""")
            return
        html(f"""<div class="hr-counts">
<div class="hr-count firm"><div class="big">{firm_n}<small>of {total}</small></div>
<div class="lbl">connection points take {size:g} MW firm: it fits even at the worst moment</div></div>
<div class="hr-count flex"><div class="big">{flex_n}<small>of {total}</small></div>
<div class="lbl">take it with a flexible connection: it fits in 95% of conditions</div></div>
<div class="hr-count"><div class="big">+{flex_n - firm_n}</div>
<div class="lbl">more points open up with flexible connections, with no new wires</div></div></div>""")
        html("""<div class="hr-note" style="margin-top:.2rem">Firm: the physics engine's capacity
search at the worst moment of the year. Flexible: the AI model's capacity over 1,000 sampled grid
conditions, then rechecked with the physics engine on 50 new ones.</div>""")


def race_markup(done, total, model_s, solver_total_s, finished):
    def clock(seconds):
        seconds = max(seconds, 0)
        h, rest = divmod(int(round(seconds)), 3600)
        m = rest // 60
        return f"{h} h {m:02d} min" if h else f"{m} min"

    share = done / total if total else 0
    solver_share = min(model_s / solver_total_s, 1) if solver_total_s else 0
    to_go = f"about {clock(solver_total_s - model_s)} to go" if solver_total_s else "waiting"
    lanes = f"""<div class="hr-lane model"><div class="top"><b>AI model</b>
<span>{done:,} of {total:,} capacities, {model_s:.1f} s</span></div>
<div class="bar"><i style="width:{share * 100:.2f}%"></i></div></div>
<div class="hr-lane solver"><div class="top"><b>Physics engine alone, same work</b>
<span>{solver_share:.2%} done, {to_go}</span></div>
<div class="bar"><i style="width:{solver_share * 100:.3f}%"></i></div></div>"""
    summary = ""
    if finished:
        summary = (f'<div class="sum"><b>{total:,} capacity searches in {model_s:.1f} s.</b> '
                   f'The physics engine alone would need about <b>{clock(solver_total_s)}</b> '
                   f'for the same job, {solver_total_s / model_s:,.0f} times longer. '
                   f'Every capacity on the maps was still rechecked with the physics engine '
                   f'before it went there.</div>')
    return clean(f'<div class="hr-race">{lanes}{summary}</div>')


def run_race(slot):
    models, features = model_and_features()
    buses = features.index.to_numpy()
    rng = np.random.default_rng(5)
    conditions = pd.DataFrame({"load_level": rng.uniform(0.10, 0.60, RACE_CONDITIONS),
                               "gen_level": rng.uniform(0.0, 0.80, RACE_CONDITIONS)})
    # The solver's time per power flow, measured now on this computer.
    probe = add_project(grid_for("Worst case"), farthest, 1.0)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        solve(probe)
        times.append(time.perf_counter() - start)
    per_solve = float(np.median(times))
    total = len(buses) * RACE_CONDITIONS
    solver_total = total * SOLVES_PER_SEARCH * per_solve

    done, model_s = 0, 0.0
    for chunk in np.array_split(np.arange(RACE_CONDITIONS), 10):
        cases = pd.DataFrame({
            "bus": np.tile(buses, len(chunk)),
            "load_level": np.repeat(conditions.load_level.to_numpy()[chunk], len(buses)),
            "gen_level": np.repeat(conditions.gen_level.to_numpy()[chunk], len(buses)),
        })
        start = time.perf_counter()
        ml.model_capacity_bisect(models, features, cases)
        model_s += time.perf_counter() - start
        done += len(cases)
        slot.markdown(race_markup(done, total, model_s, solver_total, done == total),
                      unsafe_allow_html=True)
    st.session_state.race_result = dict(done=done, total=total, model_s=model_s,
                                        solver_total_s=solver_total)


# The proof, one graph at a time: (image in results/, title, what it shows, metric key).
PROOF_GRAPHS = (
    ("proof_voltage", "Highest voltage",
     "The voltage at the grid's highest point. The shaded band is close to the safety limit (the "
     "dotted line), where a small error could turn a yes into a no.", "max_voltage_pu"),
    ("proof_line", "Busiest power line",
     "How full the busiest line is, in % of what it can carry. The AI keeps up across the whole "
     "range, including past 100%, where the line would overload.", "max_line_loading"),
    ("proof_trafo", "Busiest substation transformer",
     "The same for the transformers that link this grid to the wider network.", "max_trafo_loading"),
    ("proof_capacity", "Capacity at each connection point",
     "The number Headroom exists to find: how much new solar each point can take. The AI's search "
     "against the physics engine's own, at every point under four grid conditions.", None),
)


def proof():
    metrics_file = ml.ROOT / "results" / "model_metrics.json"
    recheck_file = ml.ROOT / "results" / "capacity95_recheck.json"
    if not metrics_file.exists():
        st.info("No test results yet: run `py -3.14 train_model.py`.")
        return
    m = json.loads(metrics_file.read_text())
    pf = m["pass_fail_model"]
    recheck = json.loads(recheck_file.read_text()) if recheck_file.exists() else None

    tiles = [
        (f"{pf['accuracy']:.1%}", f"agrees with the physics engine on whether a project fits, "
                                  f"across {m['test_scenarios']:,} cases the AI never saw"),
        (f"{pf['false_pass_rate']:.1%}", "said a project fits when the physics said no: the "
                                         "dangerous mistake"),
        (f"{m['capacity']['mean_abs_error_mw']:.2f} MW", "average error in the capacity it finds, "
                                                         "against the physics engine's own search"),
    ]
    if recheck:
        tiles.append((f"{recheck['buses_lowered']} of {recheck['buses']}",
                      f"flexible capacities lowered by the physics recheck, by "
                      f"{recheck['most_lowered_mw']:.1f} MW at most"))
    with st.container(key="proof_s1"):
        html('<div class="hr-tiles">' + "".join(
            f'<div class="hr-tile"><div class="big">{big}</div><div class="lbl">{lbl}</div></div>'
            for big, lbl in tiles) + "</div>")

    with st.container(key="proof_s2"):
        left, right = st.columns([1, 1], gap="large")
        with left, st.container(key="proof_left"):
            html('<div class="hr-h" style="margin-top:0">Why use AI at all</div>')
            html(f"""<div class="hr-copy">One capacity search takes about {SOLVES_PER_SEARCH} physics
checks. A map that holds in 95% of conditions needs that search for every connection point under a
thousand grid conditions. The AI does the same search in a fraction of the time, and the physics
engine then checks its answers. Try it: every point under {RACE_CONDITIONS} grid
conditions.</div>""")
            start = st.button(f"Run {len(mv_buses) * RACE_CONDITIONS:,} capacity searches",
                              key="race")
            slot = st.empty()
            if start:
                run_race(slot)
            elif "race_result" in st.session_state:
                r = st.session_state.race_result
                slot.markdown(race_markup(r["done"], r["total"], r["model_s"], r["solver_total_s"],
                                          True), unsafe_allow_html=True)
            else:
                slot.markdown(race_markup(0, len(mv_buses) * RACE_CONDITIONS, 0.0, 0.0, False),
                              unsafe_allow_html=True)
        with right:
            near = m.get("pass_fail_near_limit")
            near_text = (f" Close to a limit, where a small error can flip the answer, it still agrees "
                         f"{near['accuracy']:.1%} of the time ({near['cases']:,} cases)." if near else "")
            html(f"""<div class="hr-h" style="margin-top:0">How to read the graphs below</div>
<div class="hr-copy">Scroll down for one graph at a time. Each dot is a grid condition the AI never
saw: the physics engine's answer across, the AI's answer up. On the dashed line, they agree
exactly.{near_text}</div>
<div class="hr-note">The AI is never trusted alone: the physics engine rechecks every capacity
before it goes on the map.</div>""")

    for i, (name, title, text, key) in enumerate(PROOF_GRAPHS):
        path = ml.ROOT / "results" / f"{name}.png"
        if not path.exists():
            continue
        if key:
            e = m["errors"][key]
            if key == "max_voltage_pu":
                metric = (f"Average error <b>{e['mean_abs_error']:.4f} pu</b>, "
                          f"{e['mean_abs_error'] * 100:.2f}% of normal voltage; 95% of errors are "
                          f"under {e['p95_abs_error']:.4f} pu.")
            else:
                metric = (f"Average error <b>{e['mean_abs_error']:.1f} points</b> of loading; 95% of "
                          f"errors are under {e['p95_abs_error']:.1f}.")
        else:
            c = m["capacity"]
            metric = (f"Average error <b>{c['mean_abs_error_mw']:.2f} MW</b>; "
                      f"{c['share_within_0_5_mw']:.0%} of points are within 0.5 MW of the physics "
                      f"engine.")
        data = base64.b64encode(path.read_bytes()).decode()
        image = (f'<div class="hr-graph"><img alt="{title}: the AI against the physics engine, '
                 f'on cases it never saw" src="data:image/png;base64,{data}"></div>')
        words = (f'<div class="hr-h" style="margin-top:0">{title}</div>'
                 f'<div class="hr-copy">{text}</div><div class="hr-where">{metric}</div>')
        with st.container(key=f"proof_g{i}"):
            if i % 2 == 0:
                img_col, text_col = st.columns([1.3, 1], gap="large", vertical_alignment="center")
            else:
                text_col, img_col = st.columns([1, 1.3], gap="large", vertical_alignment="center")
            with img_col:
                html(image)
            with text_col:
                html(words)


def electrical_space():
    bus = int(st.session_state.bus)
    mw = round(float(st.session_state.mw), 1)
    condition = st.session_state.condition
    net, result = with_solar(bus, mw, condition)
    if result["max_voltage"] is None:  # nothing solved to draw: show the grid without it
        net, mw = with_solar(bus, 0.0, condition)[0], 0.0
    states = explore_board(net, board, bus, mw)["buses"]
    curved, flat = hyperbolic_fit()
    pairs = len(disk()["nodes"]) - 1
    pairs = pairs * (pairs - 1) // 2

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        show_disk(disk_data(net, board, disk(), states, bus, fit=(curved, flat)))
    with right, st.container(key="space_text"):
        html('<div class="hr-h" style="margin-top:0">Electrical space</div>')
        html("""<div class="hr-copy">On a map, distance is measured in kilometres. To a grid, the
distance that matters is electrical: the impedance of the lines between two points, in ohms. The
farther out a bus is electrically, the more a new solar project raises its voltage.</div>""")
        html("""<div class="hr-copy">This view draws the grid in the hyperbolic plane, where every
line's length is its impedance. A radial grid is a tree, and trees fit hyperbolic space far better
than a flat page: room grows exponentially away from any point, so every branch keeps its own
space.</div>""")
        html("""<div class="hr-copy">Drag to glide through the grid, or click a bus to travel to it.
Every distance and angle stays true as you move: the part you move to opens up, and the rest stays
in view at the edge. The faint seven-sided tiles behind the grid are all the same size in hyperbolic
space, three to a corner, which can't happen on a flat page.</div>""")
        html(f"""<div class="hr-counts" style="grid-template-columns: repeat(2, minmax(0, 1fr))">
<div class="hr-count firm"><div class="big">{curved:.0%}</div>
<div class="lbl">average distance error in this hyperbolic drawing</div></div>
<div class="hr-count"><div class="big">{flat:.0%}</div>
<div class="lbl">in a flat drawing fitted to the same distances</div></div></div>""")
        html(f"""<div class="hr-note" style="margin:.2rem 0 1rem">Measured over all {pairs:,} pairs of
buses against their true electrical distance. The flat drawing is metric multidimensional scaling
(SMACOF), which moves points to match the distances as closely as it can. Switch to Flat to see
it.</div>""")
        state_line = (f"{mw:g} MW of new solar at bus {bus}" if mw > 0 else "no new solar")
        html(f'<div class="hr-where" style="margin:0 0 1rem">Colours and moving dots show the same '
             f'physics check as Try a project: {state_line}, {condition.lower()} conditions.</div>')
        label = f"Add {DEMO_MW:g} MW of solar at bus {bus}" if mw == 0 else "Remove the solar"
        st.button(label, key="toggle_space", on_click=toggle_solar)


def from_orbit():
    if not sunlight.SUNLIGHT_FILE.exists():
        html('<div class="hr-note">No sunlight data yet: run <code>py -3.14 fetch_sunlight.py</code> '
             'once, with internet.</div>')
        return
    bus = int(st.session_state.bus)
    mw = round(float(st.session_state.mw), 1)
    real = sky_grid(file_time(sunlight.SUNLIGHT_FILE))
    clear = cloudless_grid()
    peak = float(max(real.max(), clear.max()))
    marked, mark = (), ""
    if mw > 0:
        with st.spinner("Checking every daylight hour of 2024..."):
            r = sunlight_year(bus, mw, file_time(sunlight.SUNLIGHT_FILE))
        marked = r["fail_times_utc"]
        if r["solver_fail_hours"]:
            mark = f"the {r['solver_fail_hours']:,} hours {mw:g} MW at bus {bus} would need curtailing"
    lat, lon, points, lines = grid_place()

    # 1. The two years, seen from orbit.
    with st.container(key="orbit_s1"):
        html(f"""<div class="hr-h" style="margin-top:0">2024 over the grid, as NASA's satellites measured it</div>
<div class="hr-sky orbit">{sky_axes(sky_uri(real, peak, marked), 170)}</div>
<div class="hr-h">The same year with no clouds</div>
<div class="hr-sky orbit clear">{sky_axes(sky_uri(clear, peak), 170,
                               "Every hour of 2024 over the grid if there had been no clouds")}
{sky_keys(peak, mark)}<div class="hr-note">Each column is a day and each row an hour (local
standard time). The dark streaks in the real year are clouds passing over; the cloudless year is
the Haurwitz clear-sky model.</div></div>""")

    # 2. What the clouds change.
    if has_clear and bus in caps.index:
        size = 5.0
        firm_n = int((caps.firm >= size - 1e-9).sum())
        rule_n = int((caps.rule >= size - 1e-9).sum())
        clear_n = int((caps.clear >= size - 1e-9).sum())
        row = caps.loc[bus]
        share = real.sum() / clear.sum()
        with st.container(key="orbit_s2"):
            html(f"""<div class="hr-h" style="margin-top:0">What the clouds change</div>
<div class="hr-counts">
<div class="hr-count rule"><div class="big">+{rule_n - firm_n}<small>vs +{clear_n - firm_n}</small></div>
<div class="lbl">more connection points take {size:g} MW under the 3% rule with the real sky,
against a cloudless year</div></div>
<div class="hr-count plain"><div class="big">{row.rule:.1f}<small>vs {row.clear:.1f} MW</small></div>
<div class="lbl">what bus {bus} can take under the 3% rule, real sky against cloudless</div></div>
<div class="hr-count plain"><div class="big">{share:.0%}</div>
<div class="lbl">of a cloudless year's sunlight reached the grid in 2024: {real.sum() / 1000:,.0f}
of {clear.sum() / 1000:,.0f} kWh per m²</div></div></div>
<div class="hr-note">Curtailment only bites in full sun, so the clouds are what let the 3% rule open
so many more connection points.</div>""")

    # 3. Where the data comes from.
    with st.container(key="orbit_s3"):
        globe_col, text_col = st.columns([1, 1.1], gap="large", vertical_alignment="center")
        with globe_col:
            show_globe(lat, lon)
        with text_col:
            html("""<div class="hr-h" style="margin-top:0">Where the data comes from</div>
<div class="hr-copy">NASA's CERES instruments on the Terra and Aqua satellites measure the sunlight
Earth reflects, and geostationary weather satellites track the clouds hour by hour. NASA turns both
into hourly sunlight at the ground (CERES SYN1deg, through NASA POWER).</div>
<div class="hr-copy">Drag the globe to turn it. Each time the satellite passes over the grid, the
grid lights up.</div>
<div class="hr-note">The orbit and its speed are for illustration: tilted 98° like Terra's and
Aqua's, but a real polar orbit passes over a given place about twice a day, not on every lap.</div>""")

    # 4. The grid inside one data cell.
    with st.container(key="orbit_s4"):
        text_col, cell_col = st.columns([1, 1.3], gap="large", vertical_alignment="center")
        with text_col:
            html(f"""<div class="hr-h" style="margin-top:0">One data cell covers the whole grid</div>
<div class="hr-copy">NASA's sunlight comes in cells of 1° × 1°, about 111 × 74 km here. All
{len(points)} connection points, and every line between them, sit inside one cell, so one series
of sunlight covers the whole grid.</div>
<div class="hr-note">The sunlight was downloaded once with requests, read with pandas and stored
with the app, so it runs offline. Map: Natural Earth.</div>""")
        with cell_col:
            html(f'<div class="hr-globe cell">{cell_svg(lat, lon, points, lines)}</div>')


view = st.session_state.view
intro_col, how_col = st.columns([2.4, 1], gap="large")
with intro_col:
    html(f'<div class="hr-intro">{INTROS.get(view, "")}</div>')
with how_col:
    # A popover floats over the page, so opening it doesn't push the view down.
    with st.popover("How Headroom works", width="stretch"):
        st.markdown(HOW_IT_WORKS)

# Each view in its own container, so switching views plays the entrance again.
VIEW_KEYS = {"Try a project": "try", "Capacity map": "map", "From orbit": "orbit",
             "Electrical space": "space", "Proof": "proof"}
with st.container(key=f"view_{VIEW_KEYS.get(view, 'try')}"):
    if view == "Capacity map":
        capacity_map()
    elif view == "From orbit":
        from_orbit()
    elif view == "Electrical space":
        electrical_space()
    elif view == "Proof":
        proof()
    else:
        try_a_project()

start_motion()  # last, so it runs once the page above exists
