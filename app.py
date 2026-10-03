import streamlit as st

from capacity import MAX_SEARCH_MW
from capacity_data import CAPACITY_FILE, placeholder_capacity, read_capacity
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
for key in ["mw", "view"]:
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


def show_capacity_view():
    """How much solar each bus can take, read from results/capacity.csv."""
    placeholder = not CAPACITY_FILE.exists()
    if placeholder:
        capacity = fake_capacity(tuple(mv_buses))
        st.warning(
            "**Placeholder data.** These capacities are fake, a made-up "
            "pattern, not grid results. The real ones will appear once "
            "`results/capacity.csv` exists."
        )
    else:
        try:
            capacity = capacity_from_file(CAPACITY_FILE.stat().st_mtime)
        except (KeyError, ValueError) as error:
            st.error(
                f"Could not read `results/capacity.csv` ({error}). It needs "
                "the columns `bus`, `capacity_mw` and `limit`."
            )
            return

    with st.sidebar:
        st.header("Capacity at a bus")
        st.selectbox("Bus", mv_buses, key="bus", format_func=bus_label)
        st.caption("Or click a bus on the map.")
        if selected in capacity.index:
            st.metric("Capacity", f"{capacity.capacity_mw[selected]:.1f} MW")
            limit = capacity.limit[selected]
            if limit.strip():
                st.write(f"Stopped by: {limit}")
            else:
                st.write(f"Nothing breaks up to {MAX_SEARCH_MW:g} MW.")
            if placeholder:
                st.caption("Placeholder data, not real.")
        else:
            st.write("No capacity data for this bus.")

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


page = st.radio("View", ["Project check", "Capacity"], horizontal=True, key="page")
if page == "Capacity":
    show_capacity_view()
else:
    show_project_view()
