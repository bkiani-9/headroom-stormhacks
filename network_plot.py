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
KEEP_COLOUR = dict(marker=dict(opacity=1))


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
    traces = []

    # Lines: a Plotly trace has one line colour, so lines are grouped by
    # colour, in steps of 2% of the limit (too small to see). One trace per
    # group is much faster than one per line. `None` breaks the drawing
    # between lines in the same group.
    groups = {}  # step -> (xs, ys, hover texts)
    mid_x, mid_y, mid_hover = [], [], []
    for line in net.line.index:
        coords = json.loads(net.line.geo[line])["coordinates"]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        loading = net.res_line.loading_percent[line]
        hover = (
            f"<b>{net.line.name[line]}</b> (line {line})<br>"
            f"Loading {loading:.1f}% (limit {MAX_LOADING_PERCENT:.0f}%)"
        )
        step = round(min(max(loading_used(loading), 0), 100) / 2) * 2
        gx, gy, gtext = groups.setdefault(step, ([], [], []))
        gx.extend(xs + [None])
        gy.extend(ys + [None])
        gtext.extend([hover] * len(xs) + [None])

        # An invisible point mid-line, so hovering the middle of a line works too.
        mid = len(coords) // 2
        if len(coords) % 2 == 0:
            mid_x.append((xs[mid - 1] + xs[mid]) / 2)
            mid_y.append((ys[mid - 1] + ys[mid]) / 2)
        else:
            mid_x.append(xs[mid])
            mid_y.append(ys[mid])
        mid_hover.append(hover)

    for step, (gx, gy, gtext) in sorted(groups.items()):
        traces.append(go.Scatter(
            x=gx, y=gy, mode="lines",
            line=dict(color=colour_for(step), width=2),
            hoverinfo="text", text=gtext, showlegend=False,
        ))
    traces.append(go.Scatter(
        x=mid_x, y=mid_y, mode="markers",
        marker=dict(size=12, opacity=0),
        selected=dict(marker=dict(opacity=0)),
        unselected=dict(marker=dict(opacity=0)),
        hoverinfo="text", text=mid_hover, showlegend=False,
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
    # Clicking selects a point. `KEEP_COLOUR` stops Plotly fading the rest.
    traces.append(go.Scatter(
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
        unselected=KEEP_COLOUR,
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
    traces.append(go.Scatter(
        x=subs_x, y=subs_y, mode="markers",
        marker=dict(
            symbol="square", size=18,
            color=subs_colour, colorscale=COLORSCALE, cmin=0, cmax=100,
            line=dict(color="#0b0b0b", width=2),
        ),
        unselected=KEEP_COLOUR,
        hoverinfo="text", text=subs_hover, customdata=subs_bus,
        showlegend=False,
    ))

    if selected_bus is not None:
        traces.append(ring_trace(net, selected_bus))

    # Building the figure in one go is much faster than adding traces one by one.
    return go.Figure(data=traces, layout=plain_layout(xy))


def ring_trace(net, selected_bus):
    """A ring around the selected bus, drawn on top of everything."""
    x, y = bus_xy(net, selected_bus)
    return go.Scatter(
        x=[x], y=[y], mode="markers",
        marker=dict(symbol="circle-open", size=22,
                    line=dict(color="#0b0b0b", width=3)),
        unselected=KEEP_COLOUR,
        hoverinfo="skip", customdata=[int(selected_bus)],
        showlegend=False,
    )


def plain_layout(xy):
    """Plain background, no axes, for bus positions `xy`.

    Stretches y so the grid keeps its real shape (a degree of longitude is
    shorter than a degree of latitude here).
    """
    mean_lat = sum(p[1] for p in xy) / len(xy)
    return dict(
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(
            visible=False, scaleanchor="x",
            scaleratio=1 / math.cos(math.radians(mean_lat)),
        ),
    )


# Capacity scale: red for little room, green for plenty.
# The same four colours as COLORSCALE, reversed.
CAPACITY_COLORSCALE = [
    [0.0, "#d03b3b"],
    [0.2, "#ec835a"],
    [0.5, "#fab219"],
    [1.0, "#0ca30c"],
]
NO_DATA_GREY = "#c3c2b7"


def capacity_figure(net, capacity, max_mw, selected_bus=None, placeholder=False):
    """Map of buses coloured by hosting capacity.

    `capacity` is a table indexed by bus with columns `capacity_mw` and
    `limit` (the limit that stops a bigger project; empty if none within
    `max_mw`). Buses missing from the table are drawn grey.
    Lines and substations are drawn neutral grey: only buses carry a
    capacity.
    """
    traces = []

    # All lines as one grey trace.
    lx, ly = [], []
    for line in net.line.index:
        coords = json.loads(net.line.geo[line])["coordinates"]
        lx.extend([c[0] for c in coords] + [None])
        ly.extend([c[1] for c in coords] + [None])
    traces.append(go.Scatter(
        x=lx, y=ly, mode="lines", line=dict(color=NO_DATA_GREY, width=2),
        hoverinfo="skip", showlegend=False,
    ))

    note = "<br><i>PLACEHOLDER DATA, not real</i>" if placeholder else ""
    buses = net.bus.index
    xy = [bus_xy(net, b) for b in buses]

    # Buses with a capacity, coloured; buses without, grey.
    with_data = [b for b in buses if b in capacity.index]
    without = [b for b in buses if b not in capacity.index]

    hover = []
    for b in with_data:
        mw = capacity.capacity_mw[b]
        limit = capacity.limit[b]
        if isinstance(limit, str) and limit.strip():
            stop = f"Stopped by: {limit}"
        else:
            stop = f"Nothing breaks up to {max_mw:g} MW"
        hover.append(
            f"<b>{net.bus.name[b]}</b> (bus {b})<br>"
            f"Capacity {mw:.1f} MW<br>{stop}{note}"
        )
    traces.append(go.Scatter(
        x=[bus_xy(net, b)[0] for b in with_data],
        y=[bus_xy(net, b)[1] for b in with_data],
        mode="markers",
        marker=dict(
            size=9,
            color=[capacity.capacity_mw[b] for b in with_data],
            colorscale=CAPACITY_COLORSCALE, cmin=0, cmax=max_mw,
            line=dict(color="white", width=1),
            colorbar=dict(
                title=dict(text="Capacity (MW)", side="right"),
                tickvals=[0, max_mw / 2, max_mw],
                ticktext=["0", f"{max_mw / 2:g}", f"{max_mw:g}+"],
            ),
        ),
        unselected=KEEP_COLOUR,
        hoverinfo="text", text=hover, customdata=[int(b) for b in with_data],
        showlegend=False,
    ))
    if without:
        traces.append(go.Scatter(
            x=[bus_xy(net, b)[0] for b in without],
            y=[bus_xy(net, b)[1] for b in without],
            mode="markers",
            marker=dict(size=7, color=NO_DATA_GREY),
            unselected=KEEP_COLOUR,
            hoverinfo="text",
            text=[f"<b>{net.bus.name[b]}</b> (bus {b})<br>No capacity data"
                  for b in without],
            customdata=[int(b) for b in without],
            showlegend=False,
        ))

    # Substations as grey squares; clicking one picks its 20 kV bus.
    traces.append(go.Scatter(
        x=[bus_xy(net, lv)[0] for lv in net.trafo.lv_bus],
        y=[bus_xy(net, lv)[1] for lv in net.trafo.lv_bus],
        mode="markers",
        marker=dict(symbol="square", size=18, color=NO_DATA_GREY,
                    line=dict(color="#0b0b0b", width=2)),
        unselected=KEEP_COLOUR,
        hoverinfo="text",
        text=[f"<b>{name}</b> (110/20 kV substation)" for name in net.trafo.name],
        customdata=[int(lv) for lv in net.trafo.lv_bus],
        showlegend=False,
    ))

    if selected_bus is not None:
        traces.append(ring_trace(net, selected_bus))

    return go.Figure(data=traces, layout=plain_layout(xy))
