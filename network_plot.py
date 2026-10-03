"""Draw the grid as a plain 2D plot (no map tiles, so it works offline)."""
import json
import math

import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from engine import MAX_LOADING_PERCENT, MAX_VOLTAGE_PU

# One shared scale: 0 = well within limits (green), 100 = at the limit (red).
# Colours are the status palette: good, warning, serious, critical.
COLORSCALE = [
    [0.0, "#0ca30c"],
    [0.5, "#fab219"],
    [0.8, "#ec835a"],
    [1.0, "#d03b3b"],
]
NOMINAL_VOLTAGE_PU = 1.0


def voltage_used(vm_pu):
    """How much of the room between 1.0 pu and the voltage limit is used, in %."""
    return (vm_pu - NOMINAL_VOLTAGE_PU) / (MAX_VOLTAGE_PU - NOMINAL_VOLTAGE_PU) * 100


def loading_used(loading_percent):
    """How much of the loading limit is used, in %."""
    return loading_percent / MAX_LOADING_PERCENT * 100


def colour_for(used):
    """The scale's colour for a % of limit used (clamped to 0-100)."""
    clamped = min(max(used, 0), 100) / 100
    return sample_colorscale(COLORSCALE, clamped)[0]


def bus_xy(net, bus):
    x, y = json.loads(net.bus.geo[bus])["coordinates"]
    return x, y


def network_figure(net, selected_bus=None):
    """Build the figure from a solved `net`, with a ring around `selected_bus`.

    Bus and substation points carry their bus index as `customdata`, so a
    click on the chart tells the app which bus was picked.
    """
    fig = go.Figure()

    # Lines: one trace each, so each can have its own colour.
    for line in net.line.index:
        coords = json.loads(net.line.geo[line])["coordinates"]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        loading = net.res_line.loading_percent[line]
        hover = (
            f"<b>{net.line.name[line]}</b> (line {line})<br>"
            f"Loading {loading:.1f}% (limit {MAX_LOADING_PERCENT:.0f}%)"
        )
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=colour_for(loading_used(loading)), width=2),
            hoverinfo="text", text=[hover] * len(xs),
            showlegend=False,
        ))
        # An invisible point mid-line, so hovering the middle of a line works too.
        mid = len(coords) // 2
        if len(coords) % 2 == 0:
            mx = (xs[mid - 1] + xs[mid]) / 2
            my = (ys[mid - 1] + ys[mid]) / 2
        else:
            mx, my = xs[mid], ys[mid]
        fig.add_trace(go.Scatter(
            x=[mx], y=[my], mode="markers",
            marker=dict(size=12, opacity=0),
            selected=dict(marker=dict(opacity=0)),
            unselected=dict(marker=dict(opacity=0)),
            hoverinfo="text", text=[hover], showlegend=False,
        ))

    # Buses: one trace, coloured on the shared scale, carrying the one legend.
    buses = net.bus.index
    xy = [bus_xy(net, b) for b in buses]
    vm = net.res_bus.vm_pu[buses]
    bus_hover = [
        f"<b>{net.bus.name[b]}</b> (bus {b})<br>"
        f"Voltage {vm[b]:.4f} pu (limit {MAX_VOLTAGE_PU})<br>"
        f"{max(voltage_used(vm[b]), 0):.0f}% of voltage room used"
        for b in buses
    ]
    fig.add_trace(go.Scatter(
        x=[p[0] for p in xy], y=[p[1] for p in xy], mode="markers",
        marker=dict(
            size=8,
            color=[voltage_used(v) for v in vm],
            colorscale=COLORSCALE, cmin=0, cmax=100,
            line=dict(color="white", width=1),
            colorbar=dict(
                title=dict(text="% of limit used", side="right"),
                tickvals=[0, 50, 100],
                ticktext=["0", "50", "100 (limit)"],
            ),
        ),
        hoverinfo="text", text=bus_hover, customdata=list(buses),
        showlegend=False,
    ))

    # Substations: larger squares at the 110/20 kV transformers.
    subs_x, subs_y, subs_colour, subs_hover, subs_bus = [], [], [], [], []
    for trafo in net.trafo.index:
        lv_bus = net.trafo.lv_bus[trafo]
        hv_bus = net.trafo.hv_bus[trafo]
        x, y = bus_xy(net, lv_bus)
        loading = net.res_trafo.loading_percent[trafo]
        subs_x.append(x)
        subs_y.append(y)
        subs_bus.append(int(lv_bus))  # clicking a substation picks its 20 kV bus
        subs_colour.append(loading_used(loading))
        subs_hover.append(
            f"<b>{net.trafo.name[trafo]}</b> (110/20 kV substation)<br>"
            f"Loading {loading:.1f}% (limit {MAX_LOADING_PERCENT:.0f}%)<br>"
            f"20 kV side: {net.bus.name[lv_bus]} (bus {lv_bus}), "
            f"{net.res_bus.vm_pu[lv_bus]:.4f} pu<br>"
            f"110 kV side: {net.bus.name[hv_bus]} (bus {hv_bus}), "
            f"{net.res_bus.vm_pu[hv_bus]:.4f} pu"
        )
    fig.add_trace(go.Scatter(
        x=subs_x, y=subs_y, mode="markers",
        marker=dict(
            symbol="square", size=18,
            color=subs_colour, colorscale=COLORSCALE, cmin=0, cmax=100,
            line=dict(color="#0b0b0b", width=2),
        ),
        hoverinfo="text", text=subs_hover, customdata=subs_bus,
        showlegend=False,
    ))

    # Ring around the selected bus, drawn on top of everything.
    if selected_bus is not None:
        x, y = bus_xy(net, selected_bus)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(symbol="circle-open", size=22,
                        line=dict(color="#0b0b0b", width=3)),
            hoverinfo="skip", customdata=[int(selected_bus)],
            showlegend=False,
        ))

    # Clicking selects a point; keep every other point at full colour
    # instead of letting Plotly fade them.
    fig.update_traces(
        unselected=dict(marker=dict(opacity=1)),
        selector=lambda trace: trace.customdata is not None,
    )

    # Plain background, no axes. Stretch y so the grid keeps its real shape
    # (a degree of longitude is shorter than a degree of latitude here).
    mean_lat = sum(p[1] for p in xy) / len(xy)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(
        visible=False, scaleanchor="x",
        scaleratio=1 / math.cos(math.radians(mean_lat)),
    )
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
