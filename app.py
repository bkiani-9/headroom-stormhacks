import streamlit as st

from engine import add_project, check, farthest_bus, load_grid, solve
from network_plot import network_figure


# Load and solve the grid once, then reuse it when the page reruns.
@st.cache_resource
def solved_grid():
    net = load_grid()
    solve(net)
    return net


@st.cache_resource
def far_bus():
    return farthest_bus(solved_grid())


st.title("Headroom")

net = solved_grid()
max_voltage = net.res_bus.vm_pu.max()

st.metric("Highest bus voltage (per unit)", f"{max_voltage:.4f}")

col1, col2, col3 = st.columns(3)
col1.metric("Buses", len(net.bus))
col2.metric("Lines", len(net.line))
col3.metric("Transformers", len(net.trafo))

st.subheader("The grid today")
st.caption(
    "Buses are coloured by voltage, lines by loading, substations (squares) "
    "by transformer loading, all on one scale: green is well within the "
    "limit, red is at or past it. Hover for details."
)
st.plotly_chart(network_figure(net))

st.subheader("Check a new project")

# Only the 20 kV buses: projects connect to the local grid, not the 110 kV side.
mv_buses = list(net.bus.index[net.bus.vn_kv == 20])
farthest = far_bus()


def bus_label(bus):
    if bus == farthest:
        return f"Bus {bus} (farthest from a substation)"
    return f"Bus {bus}"


bus = st.selectbox(
    "Bus", mv_buses, index=mv_buses.index(farthest), format_func=bus_label
)
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
