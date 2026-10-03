import streamlit as st

from engine import load_grid, solve


# Load and solve the grid once, then reuse it when the page reruns.
@st.cache_resource
def solved_grid():
    net = load_grid()
    solve(net)
    return net


st.title("Headroom")

net = solved_grid()
max_voltage = net.res_bus.vm_pu.max()

st.metric("Highest bus voltage (per unit)", f"{max_voltage:.4f}")

col1, col2, col3 = st.columns(3)
col1.metric("Buses", len(net.bus))
col2.metric("Lines", len(net.line))
col3.metric("Transformers", len(net.trafo))
