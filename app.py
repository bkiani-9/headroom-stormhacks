import streamlit as st

from engine import (MAX_VOLTAGE_PU, add_project, check, farthest_bus,
                    load_grid, solve)
from network_plot import network_figure, voltage_used

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

with st.sidebar:
    st.header("Selected bus")
    st.subheader(f"{net.bus.name[selected]} (bus {selected})")
    if selected == farthest:
        st.caption("Farthest bus from a substation")
    selected_vm = net.res_bus.vm_pu[selected]
    st.metric("Voltage (per unit)", f"{selected_vm:.4f}")
    st.caption(
        f"Limit {MAX_VOLTAGE_PU} pu. "
        f"{max(voltage_used(selected_vm), 0):.0f}% of the voltage room is used."
    )
    st.caption("Click a bus on the map to select it.")

st.metric("Highest bus voltage (per unit)", f"{max_voltage:.4f}")

col1, col2, col3 = st.columns(3)
col1.metric("Buses", len(net.bus))
col2.metric("Lines", len(net.line))
col3.metric("Transformers", len(net.trafo))

st.subheader("The grid today")
st.caption(
    "Buses are coloured by voltage, lines by loading, substations (squares) "
    "by transformer loading, all on one scale: green is well within the "
    "limit, red is at or past it. Hover for details, click a bus to select it."
)
st.plotly_chart(
    network_figure(net, selected),
    key=MAP_KEY, on_select="rerun", selection_mode="points",
)

st.subheader("Check a new project")


def bus_label(bus):
    label = f"{net.bus.name[bus]} (bus {bus})"
    if bus == farthest:
        label += ", farthest from a substation"
    return label


bus = st.selectbox("Bus", mv_buses, key="bus", format_func=bus_label)
mw = st.number_input("Project size (MW)", min_value=0.0, value=2.0, step=0.1)

if st.button("Check"):
    result = check(add_project(net, bus, mw))

    if result["passes"]:
        st.success(f"Pass: {mw:g} MW at bus {bus} stays within every limit.")
    else:
        st.error(f"Fail: {mw:g} MW at bus {bus}. What went wrong:")
        for problem in result["problems"]:
            st.write(f"- {problem}")

    if result["max_voltage"] is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Highest voltage (pu)", f"{result['max_voltage']:.4f}")
        c2.metric(f"Busiest line: {result['worst_line_name']}",
                  f"{result['max_line']:.1f}%")
        c3.metric(f"Busiest transformer: {result['worst_trafo_name']}",
                  f"{result['max_trafo']:.1f}%")
