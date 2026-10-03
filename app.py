import pandapower as pp
import pandapower.networks as pn
import streamlit as st


# Load and solve the grid once, then reuse it when the page reruns.
@st.cache_resource
def solve_grid():
    net = pn.mv_oberrhein("generation")
    pp.runpp(net, numba=False)
    return net


st.title("Headroom")

net = solve_grid()
max_voltage = net.res_bus.vm_pu.max()

st.metric("Highest bus voltage (per unit)", f"{max_voltage:.4f}")
