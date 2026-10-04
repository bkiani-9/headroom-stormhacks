"""Draw the grid as a plain 2D plot (no map tiles, so it works offline)."""
import json
import math

import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from engine import MAX_LOADING_PERCENT, MAX_VOLTAGE_PU

# The one colour scale used everywhere in the app: how much room is left.
# Red = no room (at or past a limit, or 0 MW), green = plenty.
# Colours are the status palette: critical, serious, warning, good.
HEADROOM_SCALE = [
    [0.0, "#d03b3b"],
    [0.2, "#ec835a"],
    [0.5, "#fab219"],
    [1.0, "#0ca30c"],
]
NOMINAL_VOLTAGE_PU = 1.0
KEEP_COLOUR = dict(marker=dict(opacity=1))
INK = "#0b0b0b"
GREY = "#c3c2b7"

# Text sizes chosen to be readable from a metre away.
FONT = dict(size=18, color=INK)
HOVER_FONT = dict(size=20)
COLORBAR_FONT = dict(size=18, color=INK)


def voltage_used(vm_pu):
    """How much of the room between 1.0 pu and the voltage limit is used, in %."""
    return (vm_pu - NOMINAL_VOLTAGE_PU) / (MAX_VOLTAGE_PU - NOMINAL_VOLTAGE_PU) * 100


def loading_used(loading_percent):
    """How much of the loading limit is used, in %."""
    return loading_percent / MAX_LOADING_PERCENT * 100


def room_left(used):
    """Room left before the limit, in % (0 at or past the limit, 100 = all)."""
    return min(max(100 - used, 0), 100)


def colour_for(room):
    """The scale's colour for a % of room left."""
    return sample_colorscale(HEADROOM_SCALE, min(max(room, 0), 100) / 100)[0]


def bus_xy(net, bus):
    x, y = json.loads(net.bus.geo[bus])["coordinates"]
    return x, y


def colorbar(title, tickvals, ticktext):
    return dict(
        title=dict(text=title, side="right", font=COLORBAR_FONT),
        tickvals=tickvals, ticktext=ticktext, tickfont=COLORBAR_FONT,
        thickness=24, len=0.8,
    )


def network_figure(net, selected_bus=None):
    """Build the figure from a solved `net`, with a ring around `selected_bus`.

    Buses are coloured by the room left under the voltage limit, lines and
    substations by the room left under their loading limit.
    Bus and substation points carry their bus index as `customdata`, so a
    click on the chart tells the app which bus was picked.
    """
    traces = []

    # Lines: a Plotly trace has one line colour, so lines are grouped by
    # colour, in steps of 2% (too small to see). One trace per group is much
    # faster than one per line. `None` breaks the drawing between lines.
    groups = {}  # room step -> (xs, ys, hover texts)
    mid_x, mid_y, mid_hover = [], [], []
    for line in net.line.index:
        coords = json.loads(net.line.geo[line])["coordinates"]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        loading = net.res_line.loading_percent[line]
        hover = f"<b>Line {line}</b><br>Loading {loading:.0f}% (limit 100%)"
        step = round(room_left(loading_used(loading)) / 2) * 2
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
            line=dict(color=colour_for(step), width=3),
            hoverinfo="text", text=gtext, showlegend=False,
        ))
    traces.append(go.Scatter(
        x=mid_x, y=mid_y, mode="markers",
        marker=dict(size=14, opacity=0),
        selected=dict(marker=dict(opacity=0)),
        unselected=dict(marker=dict(opacity=0)),
        hoverinfo="text", text=mid_hover, showlegend=False,
    ))

    # Buses: one trace on the shared scale, carrying the one colour bar.
    buses = net.bus.index
    xy = [bus_xy(net, b) for b in buses]
    vm = net.res_bus.vm_pu[buses]
    bus_hover = [f"<b>Bus {b}</b><br>Voltage {vm[b]:.3f} pu (limit {MAX_VOLTAGE_PU})"
                 for b in buses]
    traces.append(go.Scatter(
        x=[p[0] for p in xy], y=[p[1] for p in xy], mode="markers",
        marker=dict(
            size=11,
            color=[room_left(voltage_used(v)) for v in vm],
            colorscale=HEADROOM_SCALE, cmin=0, cmax=100,
            line=dict(color="white", width=1),
            colorbar=colorbar("Room left before a limit", [0, 50, 100],
                              ["none", "half", "plenty"]),
        ),
        unselected=KEEP_COLOUR,
        hoverinfo="text", text=bus_hover, customdata=list(buses),
        showlegend=False,
    ))

    # Substations: larger squares at the 110/20 kV transformers.
    subs_x, subs_y, subs_colour, subs_hover, subs_bus = [], [], [], [], []
    for trafo in net.trafo.index:
        lv_bus = net.trafo.lv_bus[trafo]
        x, y = bus_xy(net, lv_bus)
        loading = net.res_trafo.loading_percent[trafo]
        subs_x.append(x)
        subs_y.append(y)
        subs_bus.append(int(lv_bus))  # clicking a substation picks its 20 kV bus
        subs_colour.append(room_left(loading_used(loading)))
        subs_hover.append(f"<b>Substation (bus {lv_bus})</b><br>"
                          f"Transformer loading {loading:.0f}% (limit 100%)")
    traces.append(go.Scatter(
        x=subs_x, y=subs_y, mode="markers",
        marker=dict(
            symbol="square", size=24,
            color=subs_colour, colorscale=HEADROOM_SCALE, cmin=0, cmax=100,
            line=dict(color=INK, width=2),
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
        marker=dict(symbol="circle-open", size=30, line=dict(color=INK, width=4)),
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
        height=720,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="closest",
        font=FONT,
        hoverlabel=dict(font=HOVER_FONT),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(
            visible=False, scaleanchor="x",
            scaleratio=1 / math.cos(math.radians(mean_lat)),
        ),
    )


def capacity_figure(net, capacity, max_mw, selected_bus=None, placeholder=False):
    """Map of buses coloured by hosting capacity, on the same room-left scale.

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
        x=lx, y=ly, mode="lines", line=dict(color=GREY, width=3),
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
            stop = f"Stopped by {limit.replace(' (model)', '')}"
        else:
            stop = f"Nothing breaks up to {max_mw:g} MW"
        hover.append(f"<b>Bus {b}</b><br>Takes {mw:.1f} MW<br>{stop}{note}")
    traces.append(go.Scatter(
        x=[bus_xy(net, b)[0] for b in with_data],
        y=[bus_xy(net, b)[1] for b in with_data],
        mode="markers",
        marker=dict(
            size=13,
            color=[capacity.capacity_mw[b] for b in with_data],
            colorscale=HEADROOM_SCALE, cmin=0, cmax=max_mw,
            line=dict(color="white", width=1),
            colorbar=colorbar("Solar it can take", [0, max_mw / 2, max_mw],
                              ["0 MW", f"{max_mw / 2:g} MW", f"{max_mw:g}+ MW"]),
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
            marker=dict(size=9, color=GREY),
            unselected=KEEP_COLOUR,
            hoverinfo="skip",
            customdata=[int(b) for b in without],
            showlegend=False,
        ))

    # Substations as grey squares; clicking one picks its 20 kV bus.
    traces.append(go.Scatter(
        x=[bus_xy(net, lv)[0] for lv in net.trafo.lv_bus],
        y=[bus_xy(net, lv)[1] for lv in net.trafo.lv_bus],
        mode="markers",
        marker=dict(symbol="square", size=24, color=GREY, line=dict(color=INK, width=2)),
        unselected=KEEP_COLOUR,
        hoverinfo="text",
        text=[f"<b>Substation (bus {lv})</b>" for lv in net.trafo.lv_bus],
        customdata=[int(lv) for lv in net.trafo.lv_bus],
        showlegend=False,
    ))

    if selected_bus is not None:
        traces.append(ring_trace(net, selected_bus))

    return go.Figure(data=traces, layout=plain_layout(xy))
