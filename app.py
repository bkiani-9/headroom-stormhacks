import json

import streamlit as st

import ml
from capacity import MAX_SEARCH_MW
from capacity_data import (CAPACITY95_FILE, CAPACITY_FILE, HEADLINE_MW,
                           count_can_take, placeholder_capacity, read_capacity,
                           read_capacity95)
from engine import add_project, check, farthest_bus, load_grid, solve
from network_plot import capacity_figure, network_figure

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


st.set_page_config(page_title="Headroom", layout="wide")

# Styling for the large headline (plain CSS, nothing loaded from the internet).
st.markdown("""
<style>
.hl-label { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.4rem; }
.hl-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.hl-box { padding: 0.6rem 1.4rem; border: 4px solid transparent; border-radius: 12px;
          color: #52514e; }
.hl-box.hl-on { border-color: #0b0b0b; color: #0b0b0b; }
.hl-num { font-size: 4rem; font-weight: 700; line-height: 1.1; }
.hl-sub { font-size: 1.4rem; }
.proof { font-size: 1.5rem; line-height: 1.4; margin-bottom: 1rem; }
/* Less empty space at the top, so the map starts higher on the screen. */
.block-container { padding-top: 1.5rem; }
h1 { padding-top: 0; }
/* Switch and control labels large enough to read from a metre. */
[data-testid="stRadio"] label p, [data-testid="stWidgetLabel"] p { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.title("Headroom")

net = solved_grid()

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
    # Buses go by their number everywhere: pandapower's own names clash with it.
    if bus == farthest:
        return f"Bus {bus} (farthest)"
    return f"Bus {bus}"


def show_project_view(switch_slot):
    """Add solar at the selected bus and see what it does to the grid."""
    with st.sidebar:
        st.header("New solar project")
        st.selectbox("Bus", mv_buses, key="bus", format_func=bus_label)
        mw = st.slider("Solar size (MW)", 0.0, 20.0, 0.0, 0.1, key="mw")
        mw = round(mw, 1)  # 0.1 steps can arrive as 2.3000000000000003

    project_net, result = with_solar(selected, mw)
    solved = result["max_voltage"] is not None

    with st.sidebar:
        if mw == 0:
            st.markdown("Drag the slider to add solar.")
        elif result["passes"]:
            st.success(f"**Pass:** {mw:g} MW fits here.")
        else:
            st.error(f"**Fail:** {mw:g} MW breaks a limit.")
            for problem in result["problems"]:
                st.markdown(f"- {problem[0].upper()}{problem[1:]}")

    view = switch_slot.radio(
        "Map shows", ["After: with the new solar", "Before: without it"],
        horizontal=True, key="view", label_visibility="collapsed",
    )
    show_after = view.startswith("After")

    # Pick which grid to draw. If the grid with the project doesn't solve,
    # there are no new results to draw, so the map falls back to today's grid.
    if mw == 0:
        st.subheader("Today's grid")
        map_net = net
    elif not show_after:
        st.subheader(f"Before: without the {mw:g} MW")
        map_net = net
    elif solved:
        st.subheader(f"After: {mw:g} MW of solar at bus {selected}")
        map_net = project_net
    else:
        st.subheader("After: the grid does not solve (showing today's grid)")
        map_net = net

    st.plotly_chart(
        network_figure(map_net, selected),
        key=MAP_KEY, on_select="rerun", selection_mode="points",
    )


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
    boxes = []
    if saved95 is not None:
        boxes.append((f"{count_can_take(saved95, HEADLINE_MW)} of {len(saved95)}",
                      "in 95% of grid conditions", showing_95()))
    boxes.append((f"{count_can_take(saved, HEADLINE_MW)} of {len(saved)}",
                  "on today's grid", saved95 is None or not showing_95()))
    # Both numbers, large; the one the Capacity map is showing gets a border.
    html = "".join(
        f'<div class="hl-box{" hl-on" if on else ""}">'
        f'<div class="hl-num">{number}</div><div class="hl-sub">{sub}</div></div>'
        for number, sub, on in boxes
    )
    st.markdown(
        f'<div class="hl-label">{label}</div><div class="hl-row">{html}</div>',
        unsafe_allow_html=True,
    )


# The snapshot/95% toggle. It sits in the Capacity view, but the headline
# reads it too, so both switch together.
CAP_95 = "95% of conditions"
CAP_SNAPSHOT = "Snapshot (today's grid)"


def showing_95():
    return st.session_state.get("cap_map", CAP_95) == CAP_95


def show_capacity_view(saved, error, switch_slot):
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
        switch_slot.radio("Capacity", [CAP_95, CAP_SNAPSHOT], horizontal=True,
                          key="cap_map", label_visibility="collapsed")
        in_95 = showing_95()
        capacity = saved95 if in_95 else saved
    else:
        capacity = saved

    with st.sidebar:
        st.header("Capacity")
        st.selectbox("Bus", mv_buses, key="bus", format_func=bus_label)
        if selected in capacity.index:
            st.metric(f"Bus {selected} takes", f"{capacity.capacity_mw[selected]:.1f} MW")
            limit = capacity.limit[selected].replace(" (model)", "")
            if limit.strip():
                st.markdown(f"Stopped by **{limit}**")
            else:
                st.markdown(f"Nothing breaks up to {MAX_SEARCH_MW:g} MW")
            if placeholder:
                st.caption("Placeholder data, not real.")
        else:
            st.markdown("No capacity data for this bus.")

    if in_95:
        st.subheader("Solar each bus can take in 95% of grid conditions")
    else:
        st.subheader("Solar each bus can take on today's grid")
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
        f'<div class="proof">On <b>{m["test_scenarios"]:,} test scenarios it never '
        f'saw</b>, the model agreed with the real power flow <b>{pf["accuracy"]:.1%}</b> '
        f'of the time: it said pass when the solver said fail in '
        f'<b>{pf["false_pass_rate"]:.1%}</b>, and fail when the solver said pass in '
        f'<b>{pf["false_fail_rate"]:.1%}</b>.</div>',
        unsafe_allow_html=True,
    )
    st.image(str(plot_file))


# The view switch on the left; each view puts its own switch on the right,
# so both fit on one line and the map starts higher on the screen.
left, right = st.columns(2)
page = left.radio("View", ["Project check", "Capacity", "Proof"], horizontal=True,
                  key="page", label_visibility="collapsed")
if page == "Capacity":
    show_capacity_view(saved, capacity_error, right)
elif page == "Proof":
    show_proof_view()
else:
    show_project_view(right)
