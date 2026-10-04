"""Headroom, second interface: the whole grid as a live one-line diagram.

Start it with run_v2.cmd: dark theme, on port 8502, so app.py can still
run on 8501 as a fallback. Same engine, model and results files as app.py;
nothing here changes them.
"""
import base64
import json
import re
import time

import numpy as np
import pandas as pd
import streamlit as st

import ml
from capacity_data import CAPACITY95_FILE, CAPACITY_FILE, read_capacity, read_capacity95
from control_room import (GREEN, ORANGE, RED, board_layout, capacity_board, explore_board,
                          flows_outward, path_lines, show_board, substation_of)
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
    "Worst case": "Load at 10% of peak, other generators at 80%: the least room for new "
                  "solar. This is the grid as pandapower ships it.",
    "Typical": "Load at 35% of peak, other generators at 40%.",
    "Best case": "Load at 60% of peak, other generators off: the most room for new solar.",
}
VIEWS = ["Try a project", "Capacity map", "Proof"]
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


@st.cache_resource
def layout():
    return board_layout(base_grid())


@st.cache_resource
def far_bus():
    return farthest_bus(base_grid())


@st.cache_data
def capacities(firm_changed, flex_changed):
    """Firm (worst case, results/capacity.csv) and flexible (95% of
    conditions, results/capacity95.csv) capacity per bus, in one table."""
    firm = read_capacity()
    flex = read_capacity95()
    return pd.DataFrame({
        "firm": firm.capacity_mw,
        "firm_limit": firm.limit,
        "flex": np.maximum(flex.capacity_mw.reindex(firm.index).fillna(0), firm.capacity_mw),
        "pass_share": flex.pass_share.reindex(firm.index),
    })


@st.cache_resource
def model_and_features():
    return ml.load_model(), ml.read_bus_features()


def file_time(path):
    return path.stat().st_mtime if path.exists() else 0


# ---------------------------------------------------------------- look

CSS = """
<style>
:root { --ink: #FFFFFF; --ink2: #D2D2D2; --ink3: #ADADAD; --edge: #4A4A4A; --panel: #121212;
        --green: #63C132; --orange: #FFA61A; --red: #FF4D57; }
html, body, .stApp, [data-testid="stAppViewContainer"] {
  font-family: "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif; }
.stApp { background: #0A0A0A; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"] { padding: 1rem 1.5rem 1.4rem; max-width: 1720px; }
[data-testid="stVerticalBlock"] { gap: .8rem; }
[data-testid="stWidgetLabel"] p { font-size: .9rem; color: var(--ink3); }
.st-key-toggle button, .st-key-race button { border: 1.5px solid #A0A0A0; border-radius: 12px;
    background: transparent; padding: .55rem 1.4rem; min-height: 0; }
.st-key-toggle button p { font-size: 1.35rem; color: #FFFFFF; }
.st-key-race button p { font-size: 1.1rem; color: #FFFFFF; }
.st-key-toggle button:hover, .st-key-race button:hover { border-color: #FFFFFF; background: #1A1A1A; }
.hr-head { display: flex; align-items: center; gap: 14px; }
.hr-head svg { flex: none; }
.hr-name { font-size: 1.85rem; font-weight: 600; line-height: 1; color: var(--ink); }
.hr-line { font-size: .98rem; color: var(--ink3); margin-top: 5px; }
.hr-story { font-size: 1.55rem; line-height: 1.3; color: #FFFFFF; max-width: 34em; margin-bottom: .45rem; }
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
.hr-gauge { margin-top: -.45rem; }
.hr-gauge .wrap { position: relative; }
.hr-gauge .track { position: relative; height: 8px; border-radius: 4px; background: #2C2C2C; overflow: hidden; }
.hr-gauge .track i { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 4px; }
.hr-gauge .mark { position: absolute; top: -4px; width: 3px; height: 16px; border-radius: 2px;
                  background: #FFFFFF; transform: translateX(-1px); }
.hr-gauge .keys { display: flex; gap: 14px; margin-top: 7px; font-size: .84rem; color: var(--ink3); flex-wrap: wrap; }
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


def html(markup):
    """Show a piece of HTML. Line breaks and indentation become single
    spaces first, so the markdown renderer can't mistake indented HTML
    for code."""
    st.markdown(clean(markup), unsafe_allow_html=True)


def lamp_colour(value, warn_from, limit):
    if value is None or value > limit + 1e-9:
        return RED
    if value > warn_from:
        return ORANGE
    return GREEN


# ---------------------------------------------------------------- page

st.markdown(CSS, unsafe_allow_html=True)

net0 = base_grid()
board = layout()
mv_buses = [int(b) for b in net0.bus.index[net0.bus.vn_kv == 20]]
farthest = far_bus()
where = substation_of(board)
caps = capacities(file_time(CAPACITY_FILE), file_time(CAPACITY95_FILE))

# Values that live across views. Streamlit forgets a widget's value on any
# run where the widget isn't drawn, so copy them over every run.
DEFAULTS = {"view": VIEWS[0], "bus": farthest, "mw": 0.0,
            "condition": "Typical", "cap_size": MAP_SIZE_MW}
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
    st.selectbox("Bus", mv_buses, key="bus", format_func=bus_name)


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


def toggle_solar():
    """The one-click button: add the demo project, or take it away."""
    st.session_state.mw = 0.0 if st.session_state.mw > 0 else DEMO_MW


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
        return (f"{mw:g} MW at bus {bus}: the power flow finds no stable operating point. "
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
        broken.append(f"voltage goes over the {MAX_VOLTAGE_PU} limit")
    if result["max_line"] > MAX_LOADING_PERCENT:
        broken.append(f"line {result['worst_line']} overloads")
    if result["max_trafo"] > MAX_LOADING_PERCENT:
        broken.append("a substation transformer overloads")
    return f"{mw:g} MW at bus {bus}: {flow}, and {' and '.join(broken)}."


# Header: name, one line on what it does, and the view switch.
left, right = st.columns([1.5, 1], vertical_alignment="center")
with left:
    html(f"""<div class="hr-head">{LOGO}<div>
<div class="hr-name">Headroom</div>
<div class="hr-line">How much new solar each bus of a 20 kV grid can take, checked against real power flow</div>
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

    clicked = show_board(explore_board(shown, board, bus, mw if solved else 0,
                                       result["ms"] if solved else None))
    if clicked is not None and clicked != bus and clicked in mv_buses:
        st.session_state.bus = clicked
        st.rerun()

    story_col, lamp_col = st.columns([1.45, 1], gap="large")
    with story_col:
        html(f'<div class="hr-story">{story(condition, bus, mw, before_net, net, result)}</div>')
        label = f"Add {DEMO_MW:g} MW of solar at bus {bus}" if mw == 0 else "Remove the solar"
        st.button(label, key="toggle", on_click=toggle_solar)
    with lamp_col:
        if solved:
            vmax_bus = int(net.res_bus.vm_pu.idxmax())
            rows = [
                ("Highest voltage", f"bus {vmax_bus}", result["max_voltage"],
                 f"{result['max_voltage']:.4f} pu", f"limit {MAX_VOLTAGE_PU}",
                 MAX_VOLTAGE_PU - 0.01, MAX_VOLTAGE_PU),
                ("Busiest line", f"line {result['worst_line']}", result["max_line"],
                 f"{result['max_line']:.0f}%", f"limit {MAX_LOADING_PERCENT:.0f}%",
                 70.0, MAX_LOADING_PERCENT),
                ("Busiest transformer", "110/20 kV", result["max_trafo"],
                 f"{result['max_trafo']:.0f}%", f"limit {MAX_LOADING_PERCENT:.0f}%",
                 70.0, MAX_LOADING_PERCENT),
            ]
            cells = "".join(
                f'<i class="hr-lamp" style="color:{lamp_colour(v, w, lim)}"></i>'
                f'<div class="n">{name}<span>{sub}</span></div>'
                f'<div class="v">{text}<span>{limit}</span></div>'
                for name, sub, v, text, limit, w, lim in rows)
            html(f'<div class="hr-lamps">{cells}</div>')

    c1, c2, c3 = st.columns([1, 1.2, 1.2], gap="large")
    with c1:
        pick_bus()
        html(f'<div class="hr-where">{where_line(bus)}</div>')
    with c2:
        st.slider("New solar at this bus", 0.0, 20.0, step=0.1, key="mw", format="%.1f MW")
        if bus in caps.index:
            row = caps.loc[bus]
            scale = 20.0
            html(f"""<div class="hr-gauge"><div class="wrap"><div class="track">
<i style="width:{min(row.flex / scale, 1) * 100:.1f}%;background:{ORANGE}"></i>
<i style="width:{min(row.firm / scale, 1) * 100:.1f}%;background:{GREEN}"></i>
</div><div class="mark" style="left:{min(mw / scale, 1) * 100:.1f}%"></div></div>
<div class="keys"><span><i style="background:{GREEN}"></i>Firm up to {row.firm:.1f} MW</span>
<span><i style="background:{ORANGE}"></i>Flexible up to {row.flex:.1f} MW</span></div></div>""")
    with c3:
        st.segmented_control("Grid condition", list(CONDITIONS), key="condition",
                             required=True, width="stretch")
        html(f'<div class="hr-note">{CONDITION_NOTES[condition]}</div>')


def capacity_map():
    bus = int(st.session_state.bus)
    size = float(st.session_state.cap_size)

    clicked = show_board(capacity_board(base_grid(), board, size, caps.firm.to_dict(),
                                        caps.flex.to_dict(), bus))
    if clicked is not None and clicked != bus and clicked in mv_buses:
        st.session_state.bus = clicked
        st.rerun()

    firm_n = int((caps.firm >= size - 1e-9).sum())
    flex_n = int((caps.flex >= size - 1e-9).sum())
    total = len(caps)
    controls, counts = st.columns([1, 2.1], gap="large")
    with controls:
        st.slider("Solar project at every bus", 0.5, 20.0, step=0.5, key="cap_size",
                  format="%.1f MW")
        pick_bus()
        if bus in caps.index:
            row = caps.loc[bus]
            share = ("" if pd.isna(row.pass_share)
                     else f", passing {row.pass_share:.0%} of sampled conditions")
            html(f"""<div class="hr-where">Firm <b>{row.firm:.1f} MW</b>, limited by
{plain_limit(row.firm_limit)}. Flexible <b>{row.flex:.1f} MW</b>{share}.</div>""")
    with counts:
        html(f"""<div class="hr-counts">
<div class="hr-count firm"><div class="big">{firm_n}<small>of {total}</small></div>
<div class="lbl">buses take {size:g} MW firm: it fits even in the worst case</div></div>
<div class="hr-count flex"><div class="big">{flex_n}<small>of {total}</small></div>
<div class="lbl">take it with a flexible connection: it fits in 95% of conditions</div></div>
<div class="hr-count"><div class="big">+{flex_n - firm_n}</div>
<div class="lbl">buses opened up by flexibility, with no new wires</div></div></div>""")
        html("""<div class="hr-note" style="margin-top:.2rem">Firm: the solver's capacity
search on the worst-case grid. Flexible: the ML model's capacity over 1,000 sampled grid
conditions, then rechecked with the solver on 50 new ones.</div>""")


def race_markup(done, total, model_s, solver_total_s, finished):
    def clock(seconds):
        seconds = max(seconds, 0)
        h, rest = divmod(int(round(seconds)), 3600)
        m = rest // 60
        return f"{h} h {m:02d} min" if h else f"{m} min"

    share = done / total if total else 0
    solver_share = min(model_s / solver_total_s, 1) if solver_total_s else 0
    to_go = f"about {clock(solver_total_s - model_s)} to go" if solver_total_s else "waiting"
    lanes = f"""<div class="hr-lane model"><div class="top"><b>ML model</b>
<span>{done:,} of {total:,} capacities, {model_s:.1f} s</span></div>
<div class="bar"><i style="width:{share * 100:.2f}%"></i></div></div>
<div class="hr-lane solver"><div class="top"><b>Power flow alone, same work</b>
<span>{solver_share:.2%} done, {to_go}</span></div>
<div class="bar"><i style="width:{solver_share * 100:.3f}%"></i></div></div>"""
    summary = ""
    if finished:
        summary = (f'<div class="sum"><b>{total:,} capacity searches in {model_s:.1f} s.</b> '
                   f'The power flow alone would need about <b>{clock(solver_total_s)}</b> '
                   f'for the same job, {solver_total_s / model_s:,.0f} times longer. '
                   f'Every capacity on the map was still rechecked with the real power flow '
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


def proof():
    metrics_file = ml.ROOT / "results" / "model_metrics.json"
    recheck_file = ml.ROOT / "results" / "capacity95_recheck.json"
    plot_file = ml.ROOT / "results" / "model_vs_solver.png"
    if not metrics_file.exists():
        st.info("No test results yet: run `py -3.14 train_model.py`.")
        return
    m = json.loads(metrics_file.read_text())
    pf = m["pass_fail_model"]
    recheck = json.loads(recheck_file.read_text()) if recheck_file.exists() else None

    tiles = [
        (f"{pf['accuracy']:.1%}", f"pass or fail agrees with the real power flow, on "
                                  f"{m['test_scenarios']:,} scenarios the model never saw"),
        (f"{pf['false_pass_rate']:.1%}", "said pass when the power flow said fail: the "
                                         "dangerous mistake"),
        (f"{m['capacity']['mean_abs_error_mw']:.2f} MW", "average capacity error against "
                                                         "the solver's own search"),
    ]
    if recheck:
        tiles.append((f"{recheck['buses_lowered']} of {recheck['buses']}",
                      f"buses lowered by the solver recheck, by {recheck['most_lowered_mw']:.1f} MW "
                      f"at most"))
    html('<div class="hr-tiles">' + "".join(
        f'<div class="hr-tile"><div class="big">{big}</div><div class="lbl">{lbl}</div></div>'
        for big, lbl in tiles) + "</div>")

    left, right = st.columns([1, 1.05], gap="large")
    with left:
        html('<div class="hr-h">Why use a model at all</div>')
        html(f"""<div class="hr-copy">One capacity search takes about {SOLVES_PER_SEARCH} power
flows. A map that holds in 95% of conditions needs that search for every bus under a thousand
grid conditions. The model does the same search in a fraction of the time, and the solver then
checks its answers. Try it: every bus under {RACE_CONDITIONS} grid conditions.</div>""")
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
        near = m.get("pass_fail_near_limit")
        if near:
            html(f"""<div class="hr-note" style="margin-top:.2rem">Close to a limit, where a
small error can flip the answer, it still agrees {near['accuracy']:.1%} of the time
({near['cases']:,} cases). The model is never trusted alone: the solver rechecks every
capacity.</div>""")
    with right:
        if plot_file.exists():
            data = base64.b64encode(plot_file.read_bytes()).decode()
            html(f'<div class="hr-plot"><img alt="Model against solver on scenarios the model '
                 f'never saw" src="data:image/png;base64,{data}"></div>')


view = st.session_state.view
if view == "Capacity map":
    capacity_map()
elif view == "Proof":
    proof()
else:
    try_a_project()
