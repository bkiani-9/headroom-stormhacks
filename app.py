import json

import pandas as pd
import streamlit as st

import ml
from capacity import MAX_SEARCH_MW
from capacity_data import (CAPACITY95_FILE, CAPACITY_FILE, HEADLINE_MW,
                           count_can_take, placeholder_capacity, read_capacity,
                           read_capacity95)
from engine import (MAX_VOLTAGE_PU, add_project, check, farthest_bus,
                    load_grid, solve)
from network_plot import capacity_figure, network_figure, voltage_used

MAP_KEY = "grid_map"


# Load and solve the grid once, then reuse it when the page reruns.
@st.cache_resource
def solved_grid():
    net = load_grid()
    solve(net)
    return net


@st.cache_resource
def far_bus():
    return farthest_bus(solved_grid())


# Remember each (bus, size) already solved, so dragging back is instant.
@st.cache_resource(max_entries=300)
def with_solar(bus, mw):
    """The grid with `mw` of solar at `bus`, solved, and its check result."""
    project_net = add_project(solved_grid(), bus, mw)
    result = check(project_net)
    return project_net, result


# Read the capacity file once. The file's last-changed time is part of the
# cache key, so saving a new file makes the app read it again.
@st.cache_data
def capacity_from_file(last_changed):
    return read_capacity()


@st.cache_data
def capacity95_from_file(last_changed):
    return read_capacity95()


@st.cache_data
def fake_capacity(buses):
    return placeholder_capacity(buses)


# The trained model, re-loaded when the model file changes.
@st.cache_resource
def trained_model(last_changed):
    return ml.load_model(), ml.read_bus_features()


def model_or_none():
    if not ml.MODEL_FILE.exists():
        return None
    return trained_model(ml.MODEL_FILE.stat().st_mtime)


# The grid the app shows is pandapower's "generation" case.
APP_LOAD_LEVEL = 0.1
APP_GEN_LEVEL = 0.8


def clicked_bus(map_state):
    """The bus index of the clicked point on the map, or None."""
    if not map_state:
        return None
    for point in map_state["selection"]["points"]:
        data = point.get("customdata")
        if data is None:
            continue  # a line, not a bus
        if isinstance(data, list):
            data = data[0]
        return int(data)
    return None


st.title("Headroom")

net = solved_grid()
max_voltage = net.res_bus.vm_pu.max()

# Only the 20 kV buses: projects connect to the local grid, not the 110 kV side.
mv_buses = list(net.bus.index[net.bus.vn_kv == 20])
farthest = far_bus()

# One selected bus for the whole page, stored under the dropdown's key.
# A new click on the map changes it; so does picking from the dropdown.
st.session_state.setdefault("bus", farthest)
click = clicked_bus(st.session_state.get(MAP_KEY))
if click is not None and click != st.session_state.get("last_click") and click in mv_buses:
    st.session_state.bus = click
st.session_state.last_click = click
selected = st.session_state.bus

# Streamlit forgets a widget's value on any run where it isn't drawn. The
# slider and before/after switch are only drawn in one view, so copy their
# values over each run to keep them when switching views and back.
for key in ["mw", "view", "cap_map"]:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]


def bus_label(bus):
    label = f"{net.bus.name[bus]} (bus {bus})"
    if bus == farthest:
        label += ", farthest from a substation"
    return label


def show_project_view():
    """Add solar at the selected bus and see what it does to the grid."""
    with st.sidebar:
        st.header("New solar project")
        st.selectbox("Bus", mv_buses, key="bus", format_func=bus_label)
        st.caption("Or click a bus on the map.")
        mw = st.slider("Solar size (MW)", 0.0, 20.0, 0.0, 0.1, key="mw")
        mw = round(mw, 1)  # 0.1 steps can arrive as 2.3000000000000003

    project_net, result = with_solar(selected, mw)
    solved = result["max_voltage"] is not None
    _, today_result = with_solar(selected, 0.0)

    with st.sidebar:
        if result["passes"]:
            st.success(f"Pass: {mw:g} MW here stays within every limit.")
        else:
            st.error(f"Fail: {mw:g} MW here breaks a limit:")
            for problem in result["problems"]:
                st.write(f"- {problem}")

        model = model_or_none()
        if model is not None:
            models, bus_features = model
            case = pd.DataFrame([{"bus": selected, "solar_mw": mw,
                                  "load_level": APP_LOAD_LEVEL,
                                  "gen_level": APP_GEN_LEVEL}])
            guess = ml.predict(models, ml.add_features(case, bus_features))
            verdict = "pass" if ml.passes(guess).iloc[0] else "fail"
            st.caption(
                f"ML model's instant guess, no power flow: **{verdict}** "
                f"(highest voltage {guess.max_voltage_pu.iloc[0]:.4f} pu, "
                f"busiest line {guess.max_line_loading.iloc[0]:.0f}%). "
                "The solver result above is the one that counts."
            )

        today_vm = net.res_bus.vm_pu[selected]
        if solved:
            new_vm = project_net.res_bus.vm_pu[selected]
            st.metric("Voltage at this bus (per unit)", f"{new_vm:.4f}",
                      delta=f"{new_vm - today_vm:+.4f} from today",
                      delta_color="off")
            st.caption(
                f"Limit {MAX_VOLTAGE_PU} pu. "
                f"{max(voltage_used(new_vm), 0):.0f}% of the voltage room is used."
            )

    st.metric("Highest bus voltage today (per unit)", f"{max_voltage:.4f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Buses", len(net.bus))
    col2.metric("Lines", len(net.line))
    col3.metric("Transformers", len(net.trafo))

    view = st.radio(
        "Map shows", ["After: with the new solar", "Before: without it"],
        horizontal=True, key="view",
    )
    show_after = view.startswith("After")

    # Pick which grid to draw. If the grid with the project doesn't solve,
    # there are no new results to draw, so the map falls back to today's grid.
    if mw == 0:
        st.subheader("The grid today")
        st.caption("Move the solar slider to see a before and after.")
        map_net, shown = net, today_result
    elif not show_after:
        st.subheader(f"Before: the grid without the {mw:g} MW of solar")
        map_net, shown = net, today_result
    elif solved:
        st.subheader(f"After: the grid with {mw:g} MW of solar at {bus_label(selected)}")
        map_net, shown = project_net, result
    else:
        st.subheader("After: the grid does not solve with this project")
        st.caption("Showing today's grid instead.")
        map_net, shown = net, today_result

    st.caption(
        "Buses are coloured by voltage, lines by loading, substations (squares) "
        "by transformer loading, all on one scale: green is well within the "
        "limit, red is at or past it. Hover for details, click a bus to select it."
    )
    st.plotly_chart(
        network_figure(map_net, selected),
        key=MAP_KEY, on_select="rerun", selection_mode="points",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Highest voltage (pu)", f"{shown['max_voltage']:.4f}")
    c2.metric(f"Busiest line: {shown['worst_line_name']}",
              f"{shown['max_line']:.1f}%")
    c3.metric(f"Busiest transformer: {shown['worst_trafo_name']}",
              f"{shown['max_trafo']:.1f}%")


def saved_capacity():
    """The saved results as (table, error). Table is None if there's no file."""
    if not CAPACITY_FILE.exists():
        return None, None
    try:
        return capacity_from_file(CAPACITY_FILE.stat().st_mtime), None
    except (KeyError, ValueError) as error:
        return None, (
            f"Could not read `results/capacity.csv` ({error}). It needs "
            "the columns `bus`, `capacity_mw` and `limit`."
        )


def show_headline(saved):
    """How many buses can take the headline project, from the saved results."""
    if saved is None:
        st.info(
            f"The headline (how many buses can take a {HEADLINE_MW:g} MW "
            "project) appears once `results/capacity.csv` exists."
        )
        return
    label = f"Buses that can take a {HEADLINE_MW:g} MW solar project"
    if saved95 is None:
        st.metric(label, f"{count_can_take(saved, HEADLINE_MW)} of {len(saved)}")
        return
    # Both numbers; the one the Capacity map is showing gets a border.
    in_95 = showing_95()
    st.markdown(f"**{label}**")
    c1, c2 = st.columns(2)
    c1.metric("In 95% of grid conditions (solver-checked)"
              + (", shown on the map" if in_95 else ""),
              f"{count_can_take(saved95, HEADLINE_MW)} of {len(saved95)}",
              border=in_95)
    c2.metric("Snapshot: today's grid (low load, high existing solar)"
              + ("" if in_95 else ", shown on the map"),
              f"{count_can_take(saved, HEADLINE_MW)} of {len(saved)}",
              border=not in_95)


# The snapshot/95% toggle. It sits in the Capacity view, but the headline
# reads it too, so both switch together.
CAP_95 = "95% of conditions"
CAP_SNAPSHOT = "Snapshot (today's grid)"


def showing_95():
    return st.session_state.get("cap_map", CAP_95) == CAP_95


def show_capacity_view(saved, error):
    """How much solar each bus can take, read from results/capacity.csv
    (today's grid) or results/capacity95.csv (95% of conditions)."""
    if error:
        st.error(error)
        return
    placeholder = saved is None
    in_95 = False
    if placeholder:
        capacity = fake_capacity(tuple(mv_buses))
        st.warning(
            "**Placeholder data.** These capacities are fake, a made-up "
            "pattern, not grid results. The real ones will appear once "
            "`results/capacity.csv` exists."
        )
    elif saved95 is not None:
        # Changing this reruns the page, so the headline above switches too.
        st.radio("Capacity", [CAP_95, CAP_SNAPSHOT], horizontal=True, key="cap_map")
        in_95 = showing_95()
        capacity = saved95 if in_95 else saved
    else:
        capacity = saved

    with st.sidebar:
        st.header("Capacity at a bus")
        st.selectbox("Bus", mv_buses, key="bus", format_func=bus_label)
        st.caption("Or click a bus on the map.")
        if selected in capacity.index:
            st.metric("Capacity in 95% of conditions" if in_95 else "Capacity",
                      f"{capacity.capacity_mw[selected]:.1f} MW")
            limit = capacity.limit[selected]
            if limit.strip():
                st.write(f"{'Most often stopped' if in_95 else 'Stopped'} by: {limit}")
            else:
                st.write(f"Nothing breaks up to {MAX_SEARCH_MW:g} MW.")
            if in_95:
                st.caption(
                    f"At this size, {capacity.pass_share[selected]:.0%} of 50 new "
                    "grid conditions pass, checked with the real power flow."
                )
            if placeholder:
                st.caption("Placeholder data, not real.")
        else:
            st.write("No capacity data for this bus.")

    if in_95:
        st.subheader("How much solar each bus can take in 95% of grid conditions")
        st.caption(
            "Load 10-60% and existing solar 0-80%. The ML model finds the size "
            "each bus takes in 95% of 1,000 sampled conditions; the real power "
            "flow then rechecks it on 50 new conditions, lowering it 0.1 MW at a "
            "time wherever more than 5% fail. Red is little room, green is plenty."
        )
    else:
        st.subheader("How much solar each bus can take")
        st.caption(
            "Each bus is coloured by its capacity: the largest solar project it "
            "takes without breaking a limit. Red is little room, green is plenty. "
            "Hover for the capacity and the limit that stops it."
        )
    st.plotly_chart(
        capacity_figure(net, capacity, MAX_SEARCH_MW, selected, placeholder),
        key=MAP_KEY, on_select="rerun", selection_mode="points",
    )


saved, capacity_error = saved_capacity()
saved95 = (capacity95_from_file(CAPACITY95_FILE.stat().st_mtime)
           if CAPACITY95_FILE.exists() else None)
show_headline(saved)

def show_proof_view():
    """Proof that the ML model matches the real solver: one sentence and
    the model-versus-solver plot, both read from files train_model.py saved."""
    st.subheader("Proof: the ML model against the real power flow")
    metrics_file = ml.ROOT / "results" / "model_metrics.json"
    plot_file = ml.ROOT / "results" / "model_vs_solver.png"
    if not metrics_file.exists() or not plot_file.exists():
        st.info("No test results yet: run `py -3.14 train_model.py`.")
        return
    m = json.loads(metrics_file.read_text())
    pf = m["pass_fail_model"]
    st.markdown(
        f"On **{m['test_scenarios']:,} test scenarios it never saw**, the model "
        f"agreed with the real power flow **{pf['accuracy']:.1%}** of the time: "
        f"it said pass when the solver said fail in **{pf['false_pass_rate']:.1%}**, "
        f"and fail when the solver said pass in **{pf['false_fail_rate']:.1%}**."
    )
    st.image(str(plot_file),
             caption="Each dot is a test scenario: solver value across, model "
                     "value up. On the dashed line the model is exactly right.")

    cap = m["capacity"]
    with st.expander("More detail"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Pass/fail agrees, close to a limit",
                  f"{m['pass_fail_near_limit']['accuracy']:.1%}")
        c2.metric("Capacity within 0.5 MW of the search", f"{cap['share_within_0_5_mw']:.0%}")
        c3.metric("Trained on", f"{m['train_scenarios']:,} scenarios")
        report = ml.ROOT / "results" / "model_report.md"
        if report.exists():
            st.markdown(report.read_text(encoding="utf-8"))


page = st.radio("View", ["Project check", "Capacity", "Proof"], horizontal=True, key="page")
if page == "Capacity":
    show_capacity_view(saved, capacity_error)
elif page == "Proof":
    show_proof_view()
else:
    show_project_view()
