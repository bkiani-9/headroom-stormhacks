"""The board for app_v2.py: the whole grid as a clean one-line diagram.

Each substation is a large rounded tile on the left. Its feeders run to
the right, one bus per tile; a side branch leaves a bus straight up or
down, then runs right again. Lines are coloured by how loaded they are,
white dots move along them in the real direction of active power (faster
for more MW), and the bar inside each bus shows how much of the room under
the voltage limit is used.

Everything is inline HTML, CSS and JavaScript in a Streamlit component
(st.components.v2): nothing is loaded from the internet, so it works
offline. diagram.py and app.py are not used or changed.
"""
import networkx as nx
import pandapower.topology as top
import streamlit as st

from engine import MAX_LOADING_PERCENT, MAX_VOLTAGE_PU

GREEN = "#63C132"
ORANGE = "#FFA61A"
RED = "#FF4D57"
LINE_DIM = "#333333"
BUSY_FROM = 70.0  # line loading (%) from which a line shows orange
WARN_BAND_PU = 0.01  # a bus within this of the voltage limit shows orange
NOMINAL_PU = 1.0


def board_layout(net):
    """A (row, column) cell for every 20 kV bus.

    The grid is radial, so each substation's area is a tree. A feeder runs
    along one row, always continuing with its longest branch; a bus's other
    branch starts in the same column, one or more rows above or below, and
    runs right from there. Branches are placed from the far end of a feeder
    back toward the substation, in the nearest free rows; if none is free,
    a new row is opened. Lines that close a loop (normally-open points) are
    not part of the tree and are not drawn.
    """
    graph = top.create_nxgraph(net, include_trafos=False, respect_switches=True)
    subs, buses, edges = [], [], []
    offset = 0
    for number, root in enumerate(net.trafo.lv_bus):
        root = int(root)
        tree = nx.bfs_tree(graph, root)
        height = {}  # longest path below each bus, to pick the straight branch
        for bus in reversed(list(nx.topological_sort(tree))):
            height[bus] = 1 + max((height[c] for c in tree.successors(bus)), default=0)

        cells = {}  # (row, col) -> bus, or "|" where a vertical line passes
        pos = {}
        drops = []  # (parent, child) joined by a vertical line

        def chain_from(bus):
            chain = [bus]
            while True:
                kids = list(tree.successors(chain[-1]))
                if not kids:
                    return chain
                chain.append(max(kids, key=lambda c: (height[c], -c)))

        def redraw_drops():
            for key in [k for k, v in cells.items() if v == "|"]:
                del cells[key]
            for parent, child in drops:
                (r0, col), (r1, _) = pos[parent], pos[child]
                for r in range(min(r0, r1) + 1, max(r0, r1)):
                    cells[(r, col)] = "|"

        def open_row_below(row):
            for bus, (r, c) in list(pos.items()):
                if r > row:
                    pos[bus] = (r + 1, c)
            cells.clear()
            for bus, (r, c) in pos.items():
                cells[(r, c)] = bus
            redraw_drops()

        def fits(chain, col0, row, prow=None):
            for c in range(col0 - 1, col0 + len(chain) + 1):  # keep a free cell at each end
                if (row, c) in cells:
                    return False
            if prow is not None:
                for r in range(min(prow, row) + 1, max(prow, row)):
                    if (r, col0) in cells:
                        return False
            return True

        def place(chain, col0, row, parent):
            for i, bus in enumerate(chain):
                cells[(row, col0 + i)] = bus
                pos[bus] = (row, col0 + i)
            previous = parent
            for bus in chain:
                edges.append({"parent": int(previous), "child": int(bus),
                              "line": _line_between(graph, previous, bus)})
                previous = bus
            if parent != root:
                drops.append((parent, chain[0]))
                redraw_drops()
            # Branches, from the far end of this chain back to its start.
            for i in reversed(range(len(chain) - 1)):
                bus, straight_on = chain[i], chain[i + 1]
                for kid in sorted(c for c in tree.successors(bus) if c != straight_on):
                    branch = chain_from(kid)
                    prow, pcol = pos[bus]
                    target = None
                    for step in range(1, 7):
                        target = next((r for r in (prow + step, prow - step)
                                       if fits(branch, pcol, r, prow)), None)
                        if target is not None:
                            break
                    if target is None:
                        open_row_below(prow)
                        prow, pcol = pos[bus]
                        target = prow + 1
                    place(branch, pcol, target, bus)

        feeders = sorted(tree.successors(root), key=lambda c: (-height[c], c))
        for k, kid in enumerate(feeders):
            feeder = chain_from(kid)
            row = 0
            if k > 0:  # the highest free row below everything placed so far
                row = max(r for r, _ in cells) + 1
                while row > 1 and fits(feeder, 1, row - 1):
                    row -= 1
            place(feeder, 1, row, root)

        top_row = min(r for r, _ in pos.values())
        bottom_row = max(r for r, _ in pos.values())
        feeder_rows = [pos[c][0] for c in feeders]
        for bus, (r, c) in pos.items():
            buses.append({"id": int(bus), "row": r - top_row + offset, "col": c})
        subs.append({"bus": root, "label": f"Substation {number + 1}",
                     "first": min(feeder_rows) - top_row + offset,
                     "last": max(feeder_rows) - top_row + offset,
                     "top": offset, "bottom": bottom_row - top_row + offset})
        offset += bottom_row - top_row + 2  # one empty row between substation areas

    return {"id": f"board-{len(buses)}-{len(edges)}-{offset}",
            "rows": offset - 1, "cols": max(b["col"] for b in buses),
            "subs": subs, "buses": buses, "edges": edges}


def _line_between(graph, a, b):
    keys = [k for k in graph.get_edge_data(a, b) if k[0] == "line"]
    return int(keys[0][1])


def substation_of(layout):
    """Map bus -> (substation label, number of lines from it)."""
    parent = {e["child"]: e["parent"] for e in layout["edges"]}
    label = {s["bus"]: s["label"] for s in layout["subs"]}
    out = {bus: (name, 0) for bus, name in label.items()}
    for b in layout["buses"]:
        hops, here = 0, b["id"]
        while here not in label:
            here = parent[here]
            hops += 1
        out[b["id"]] = (label[here], hops)
    return out


def path_lines(layout, bus):
    """The lines from the substation out to `bus`, as (line, parent, child)."""
    parent = {e["child"]: (e["parent"], e["line"]) for e in layout["edges"]}
    path = []
    while bus in parent:
        up, line = parent[bus]
        path.append((line, up, bus))
        bus = up
    return path[::-1]


def flows_outward(net, path):
    """For each line on `path`, True if power flows away from the substation."""
    out = []
    for line, up, _ in path:
        p = float(net.res_line.p_from_mw[line])
        sending = int(net.line.from_bus[line] if p >= 0 else net.line.to_bus[line])
        out.append(sending == up)
    return out


def _line_colour(loading):
    if loading > MAX_LOADING_PERCENT:
        return RED
    if loading > BUSY_FROM:
        return ORANGE
    return GREEN


def _flows(net, layout, dim=False):
    """Direction, size and colour of active power on every drawn line."""
    out = []
    for e in layout["edges"]:
        line = e["line"]
        p = float(net.res_line.p_from_mw[line])
        loading = float(net.res_line.loading_percent[line])
        sending = int(net.line.from_bus[line] if p >= 0 else net.line.to_bus[line])
        outward = sending == e["parent"]
        way = "away from the substation" if outward else "toward the substation"
        out.append({
            "dir": 1 if outward else -1,
            "mw": round(abs(p), 3),
            "stroke": LINE_DIM if dim else _line_colour(loading),
            "title": f"Line {line}",
            "detail": f"{abs(p):.2f} MW {way}, loaded {loading:.0f}%",
        })
    return out


def _substations(net, layout):
    out = []
    for s in layout["subs"]:
        trafo = int(net.trafo.index[net.trafo.lv_bus == s["bus"]][0])
        p = float(net.res_trafo.p_hv_mw[trafo])  # + means power comes down from 110 kV
        loading = float(net.res_trafo.loading_percent[trafo])
        rating = float(net.trafo.sn_mva[trafo])
        if p >= 0:
            flow = f"taking {p:.1f} MW from the 110 kV grid"
        else:
            flow = f"sending {-p:.1f} MW back up to the 110 kV grid"
        out.append({
            "note": f"{loading:.0f}% loaded",
            "over": loading > MAX_LOADING_PERCENT,
            "title": f"{s['label']} (bus {s['bus']})",
            "detail": f"{rating:.0f} MVA transformer, {loading:.0f}% loaded, {flow}",
        })
    return out


def _voltage_bus(vm):
    """Bar length, bar colour and state for a bus voltage."""
    used = (vm - NOMINAL_PU) / (MAX_VOLTAGE_PU - NOMINAL_PU)
    if vm > MAX_VOLTAGE_PU + 1e-9:
        return 1.0, RED, "over"
    if vm > MAX_VOLTAGE_PU - WARN_BAND_PU:
        return used, ORANGE, "warn"
    return used, GREEN, "idle"


def explore_board(net, layout, selected, mw, solve_ms=None):
    """Board data for Try a project: live flows, and in each bus a bar of
    how much room under the voltage limit is used."""
    where = substation_of(layout)
    buses = []
    for b in layout["buses"]:
        vm = float(net.res_bus.vm_pu[b["id"]])
        used, colour, state = _voltage_bus(vm)
        sub, hops = where[b["id"]]
        buses.append({"bar": round(min(max(used, 0.0), 1.0), 3), "colour": colour,
                      "state": state, "title": f"Bus {b['id']}",
                      "detail": f"Voltage {vm:.4f} pu (limit {MAX_VOLTAGE_PU}), "
                                f"{hops} lines from {sub}"})
    legend = [
        {"line": GREEN, "label": f"Line under {BUSY_FROM:.0f}% loaded"},
        {"line": ORANGE, "label": f"{BUSY_FROM:.0f} to {MAX_LOADING_PERCENT:.0f}%"},
        {"line": RED, "label": "Overloaded"},
        {"bar": True, "label": f"Bar in a bus: voltage room used, red over {MAX_VOLTAGE_PU} pu"},
        {"dots": True, "label": "Power flow, faster for more MW"},
    ]
    foot = f"AC power flow: {solve_ms:.0f} ms" if solve_ms else ""
    label = f"Bus {selected}" + (f": +{mw:g} MW solar" if mw > 0 else "")
    return {"layout": layout, "mode": "explore", "selected": int(selected), "label": label,
            "buses": buses, "lines": _flows(net, layout), "subs": _substations(net, layout),
            "legend": legend, "foot": foot}


def capacity_board(net, layout, size, firm, flexible, selected):
    """Board data for the Capacity map: each bus outlined green (takes
    `size` MW firm, even in the worst case), orange (only with a flexible
    connection, in 95% of conditions) or red (can't). The bar in each bus
    shows firm (green) and flexible (orange) capacity as a share of `size`.
    `firm` and `flexible` map bus -> MW."""
    where = substation_of(layout)
    buses = []
    for b in layout["buses"]:
        f = float(firm.get(b["id"], 0.0))
        x = max(float(flexible.get(b["id"], 0.0)), f)
        if size <= f + 1e-9:
            state, word = "firm", "Firm"
        elif size <= x + 1e-9:
            state, word = "flex", "Flexible connection"
        else:
            state, word = "cant", "Can't take it"
        sub, hops = where[b["id"]]
        buses.append({"bar": round(min(f / size, 1.0), 3), "bar2": round(min(x / size, 1.0), 3),
                      "colour": GREEN, "state": state,
                      "title": f"Bus {b['id']}: {word}",
                      "detail": f"Firm {f:.1f} MW, flexible {x:.1f} MW, {hops} lines from {sub}"})
    legend = [
        {"tile": GREEN, "label": "Firm: fits even in the worst case"},
        {"tile": ORANGE, "label": "Flexible: fits in 95% of conditions"},
        {"tile": RED, "label": "Can't take it"},
        {"bar": True, "label": "Bar: firm and flexible capacity against the project size"},
    ]
    return {"layout": layout, "mode": "capacity", "selected": int(selected),
            "label": f"Bus {selected}", "buses": buses, "lines": _flows(net, layout, dim=True),
            "subs": _substations(net, layout), "legend": legend,
            "foot": f"{size:g} MW at every bus"}


CSS = """
:host { display: block; }
.hb { --ink: #FFFFFF; --ink2: #D2D2D2; --ink3: #ADADAD; --tile: #161616; --edge: #A0A0A0;
      --lag: 0ms; position: relative; color: var(--ink);
      font-family: "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif;
      font-variant-numeric: tabular-nums; user-select: none; }
.hb.mode-capacity { --lag: 12ms; }
.hb svg { display: block; width: 100%; height: auto; overflow: visible; }
.hb text { font-family: inherit; }
.ln { fill: none; stroke-width: 4.6; stroke-linecap: round; transition: stroke .45s ease; }
.dots { fill: none; stroke: #FFFFFF; stroke-width: 5.8; stroke-linecap: round;
        stroke-dasharray: 0 15; pointer-events: none; transition: opacity .4s ease; }
.hit-line { fill: none; stroke: transparent; stroke-width: 14; cursor: help; }
.bus { cursor: pointer; }
.bus .tile { fill: var(--tile); stroke: var(--edge); stroke-width: 1.8;
             transition: stroke .4s ease, fill .4s ease; transition-delay: calc(var(--d) * var(--lag)); }
.bus .track { fill: #383838; pointer-events: none; }
.bus .bar, .bus .bar2 { pointer-events: none; transition: width .45s ease, fill .45s ease;
                        transition-delay: calc(var(--d) * var(--lag)); }
.bus:hover .tile { stroke: #FFFFFF; }
.bus.sel .tile { stroke: #FFFFFF; stroke-width: 2.8; filter: drop-shadow(0 0 5px rgba(255,255,255,.5)); }
.bus.over .tile { stroke: #FF4D57; stroke-width: 2.6; filter: drop-shadow(0 0 7px rgba(255,77,87,.8)); }
.bus.firm .tile { stroke: #63C132; fill: #12230B; }
.bus.flex .tile { stroke: #FFA61A; fill: #2E1E05; }
.bus.cant .tile { stroke: #FF4D57; fill: #33100F; }
.bus.sel.firm .tile, .bus.sel.flex .tile, .bus.sel.cant .tile { stroke-width: 3; }
.sub rect { fill: var(--tile); stroke: #A6A6A6; stroke-width: 1.8; }
.sub.over rect { stroke: #FF4D57; }
.sub .name { fill: #FFFFFF; font-size: 15px; font-weight: 600; text-anchor: middle; }
.sub .note { fill: var(--ink3); font-size: 13px; text-anchor: middle; }
.sub { cursor: help; }
.label rect { fill: #0A0A0A; }
.label text { fill: #FFFFFF; font-size: 13.5px; text-anchor: middle; }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 18px;
          padding: 2px 4px 0; font-size: 13px; color: var(--ink3); }
.legend i { display: inline-block; margin-right: 7px; vertical-align: middle; }
.legend .ln-sw { width: 22px; height: 4px; border-radius: 2px; }
.legend .tile-sw { width: 13px; height: 13px; border-radius: 4px; border: 1.8px solid; background: #161616; }
.legend .bar-sw { width: 22px; height: 5px; border-radius: 3px;
                  background: linear-gradient(90deg, #63C132 0 45%, #FFA61A 45% 70%, #383838 70%); }
.legend .dot-sw { width: 26px; height: 6px;
                  background: radial-gradient(circle, #fff 0 2.4px, transparent 2.8px) 0 50% / 9px 6px repeat-x;
                  animation: hb-legend .9s linear infinite; }
@keyframes hb-legend { to { background-position: 9px 50%; } }
.legend .foot { margin-left: auto; }
.tip { position: absolute; pointer-events: none; z-index: 3; opacity: 0;
       background: rgba(22,22,22,.97); border: 1px solid #6E6E6E; border-radius: 9px;
       padding: 7px 11px; font-size: 13.5px; line-height: 1.35; color: var(--ink);
       box-shadow: 0 10px 24px -10px rgba(0,0,0,.9); max-width: 320px; transition: opacity .12s ease; }
.tip b { display: block; font-weight: 600; font-size: 14.5px; }
.clock { position: absolute; bottom: 34px; right: 8px; z-index: 2; opacity: 0; pointer-events: none;
         background: rgba(10,10,10,.93); border: 1px solid #5E5E5E; border-radius: 14px;
         padding: 9px 14px 10px; width: 196px; box-sizing: border-box; transition: opacity .3s ease; }
.hb.playing .clock { opacity: 1; }
.clock .time { font-size: 36px; font-weight: 600; line-height: 1; color: #FFFFFF; }
.clock .row { display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size: 12.5px; color: var(--ink3); }
.clock .sun { flex: 1; height: 6px; border-radius: 3px; background: #2C2C2C; overflow: hidden; }
.clock .sun i { display: block; height: 100%; width: 0; background: #FFD34D; border-radius: 3px;
                box-shadow: 0 0 8px rgba(255,211,77,.8); transition: width .5s ease; }
.clock .status { margin-top: 8px; font-size: 14px; font-weight: 600; color: #7EDB4C; }
.clock.alarm .time, .clock.alarm .status { color: #FF4D57; }
.clock .what { margin-top: 4px; font-size: 12px; line-height: 1.3; color: var(--ink3); }
.tip span { color: var(--ink2); }
@media (prefers-reduced-motion: reduce) {
  .legend .dot-sw { animation: none; }
  .bus .tile, .bus .bar, .bus .bar2 { transition: none; }
}
"""

JS = r"""
const NS = "http://www.w3.org/2000/svg";
const CW = 46, RH = 46, S = 26, SUBW = 118, X1 = SUBW + 52, TOP = 22, PADR = 20, PADB = 30;
const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
const xOf = (col) => X1 + (col - 1) * CW;
const yOf = (row) => TOP + S / 2 + row * RH;

function build(host, data) {
  host.querySelectorAll(".hb").forEach((n) => n.remove());
  const L = data.layout;
  const W = xOf(L.cols) + S / 2 + PADR;
  const H = yOf(L.rows - 1) + S / 2 + PADB;
  const wrap = document.createElement("div");
  wrap.className = "hb";
  host.appendChild(wrap);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                          "aria-label": "One-line diagram of the grid" }, wrap);
  const legend = document.createElement("div");
  legend.className = "legend";
  wrap.appendChild(legend);
  const tip = document.createElement("div");
  tip.className = "tip";
  wrap.appendChild(tip);
  const clock = document.createElement("div");
  clock.className = "clock";
  clock.innerHTML = '<div class="time"></div><div class="row"><span>Sun</span>' +
    '<div class="sun"><i></i></div></div><div class="status"></div><div class="what"></div>';
  wrap.appendChild(clock);

  const state = { layoutId: L.id, wrap, svg, tip, legend, buses: {}, edges: [], subs: [], W, H,
                  clock, clockTime: clock.querySelector(".time"),
                  clockSun: clock.querySelector(".sun i"),
                  clockStatus: clock.querySelector(".status"),
                  clockWhat: clock.querySelector(".what") };
  const showTip = (evt, info) => {
    if (!info) return;
    const box = wrap.getBoundingClientRect();
    tip.replaceChildren();
    const b = document.createElement("b"); b.textContent = info.title; tip.appendChild(b);
    const s = document.createElement("span"); s.textContent = info.detail; tip.appendChild(s);
    tip.style.opacity = 1;
    let x = evt.clientX - box.left + 16, y = evt.clientY - box.top + 16;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    if (x + w > box.width - 4) x = evt.clientX - box.left - w - 14;
    if (y + h > box.height - 4) y = evt.clientY - box.top - h - 12;
    tip.style.left = `${x}px`; tip.style.top = `${y}px`;
  };
  const hideTip = () => { tip.style.opacity = 0; };

  const gLines = el("g", {}, svg), gDots = el("g", {}, svg), gHit = el("g", {}, svg);
  const gSubs = el("g", {}, svg), gBus = el("g", {}, svg), gMark = el("g", {}, svg);

  const pos = {};
  L.buses.forEach((b) => { pos[b.id] = { x: xOf(b.col), y: yOf(b.row) }; });
  const subOf = {};
  L.subs.forEach((s) => { subOf[s.bus] = s; });

  // Substations: a large rounded tile spanning the rows its feeders leave from.
  L.subs.forEach((s, i) => {
    const y0 = yOf(s.first) - S / 2 - 10, y1 = yOf(s.last) + S / 2 + 10;
    const h = Math.max(y1 - y0, 74), yc = (y0 + y1) / 2;
    const g = el("g", { class: "sub" }, gSubs);
    el("rect", { x: 6, y: yc - h / 2, width: SUBW, height: h, rx: 13 }, g);
    const name = el("text", { x: 6 + SUBW / 2, y: yc - 2, class: "name" }, g);
    name.textContent = s.label;
    const note = el("text", { x: 6 + SUBW / 2, y: yc + 18, class: "note" }, g);
    g.addEventListener("mousemove", (evt) => showTip(evt, state.subInfo && state.subInfo[i]));
    g.addEventListener("mouseleave", hideTip);
    state.subs.push({ g, note });
  });

  // Lines run centre to centre; the tiles drawn on top hide the ends, so the
  // dots seem to pass through each bus.
  L.edges.forEach((e, i) => {
    const b = pos[e.child];
    const a = subOf[e.parent] ? { x: 6 + SUBW, y: b.y } : pos[e.parent];
    const d = `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
    const ln = el("path", { d, class: "ln" }, gLines);
    const dots = el("path", { d, class: "dots" }, gDots);
    const hit = el("path", { d, class: "hit-line" }, gHit);
    hit.addEventListener("mousemove", (evt) => showTip(evt, state.lineInfo && state.lineInfo[i]));
    hit.addEventListener("mouseleave", hideTip);
    state.edges.push({ ln, dots, phase: Math.random() * 15, speed: 0, target: 0 });
  });

  // Buses: uniform rounded tiles, each with a small bar near the bottom.
  const BW = S - 10;
  L.buses.forEach((b, i) => {
    const p = pos[b.id];
    const g = el("g", { class: "bus", style: `--d:${b.col}` }, gBus);
    el("rect", { x: p.x - S / 2 - 6, y: p.y - S / 2 - 6, width: S + 12, height: S + 12,
                 fill: "transparent" }, g);
    el("rect", { class: "tile", x: p.x - S / 2, y: p.y - S / 2, width: S, height: S, rx: 6 }, g);
    el("rect", { class: "track", x: p.x - BW / 2, y: p.y + S / 2 - 8.5, width: BW, height: 4, rx: 2 }, g);
    const bar2 = el("rect", { class: "bar2", x: p.x - BW / 2, y: p.y + S / 2 - 8.5, height: 4, rx: 2,
                              width: 0, fill: "#FFA61A" }, g);
    const bar = el("rect", { class: "bar", x: p.x - BW / 2, y: p.y + S / 2 - 8.5, height: 4, rx: 2,
                             width: 0, fill: "#63C132" }, g);
    g.addEventListener("click", () => state.select(b.id));
    g.addEventListener("mousemove", (evt) => showTip(evt, state.busInfo && state.busInfo[i]));
    g.addEventListener("mouseleave", hideTip);
    state.buses[b.id] = { g, bar, bar2 };
  });
  state.BW = BW;

  // The selected bus's name, on a dark pill so lines never run through it.
  state.label = el("g", { class: "label" }, gMark);
  state.labelBox = el("rect", { rx: 6, height: 19 }, state.label);
  state.labelText = el("text", {}, state.label);
  state.pos = pos;
  state.select = (id) => {
    hideTip();
    mark(state, id, `Bus ${id}`);
    if (state.setTrigger) state.setTrigger("selected", id);
  };
  return state;
}

function mark(state, id, text) {
  for (const k in state.buses) state.buses[k].g.classList.toggle("sel", Number(k) === id);
  const p = state.pos[id];
  state.label.setAttribute("visibility", p ? "visible" : "hidden");
  if (!p) return;
  state.labelText.textContent = text;
  const w = Math.max(56, text.length * 7.3 + 14);
  const x = Math.min(Math.max(p.x - w / 2, 4), state.W - w - 4);
  const y = p.y + S / 2 + 5;
  state.labelBox.setAttribute("x", x);
  state.labelBox.setAttribute("y", y);
  state.labelBox.setAttribute("width", w);
  state.labelText.setAttribute("x", x + w / 2);
  state.labelText.setAttribute("y", y + 14);
}

// Colours, bars and flows: everything that changes from one grid state to the next.
function paint(state, layout, buses, lines, subs, capacity) {
  lines.forEach((f, i) => {
    const e = state.edges[i];
    e.ln.style.stroke = f.stroke;
    const moving = f.mw >= 0.02;
    // Speed in drawing units per second: grows with the square root of MW.
    e.target = moving ? f.dir * (10 + 26 * Math.sqrt(f.mw)) : 0;
    e.dots.style.opacity = moving ? (capacity ? 0.35 : 0.95) : 0;
  });
  subs.forEach((f, i) => {
    state.subs[i].note.textContent = f.note;
    state.subs[i].g.classList.toggle("over", !!f.over);
  });
  layout.buses.forEach((b, i) => {
    const info = buses[i], node = state.buses[b.id];
    for (const k of ["warn", "over", "firm", "flex", "cant", "idle"]) {
      node.g.classList.toggle(k, info.state === k);
    }
    // Always show at least a dot, like a gauge needle resting at zero.
    node.bar.setAttribute("width", Math.max(3.5, state.BW * info.bar));
    node.bar.setAttribute("fill", info.colour);
    const two = info.bar2 !== undefined;
    node.bar2.setAttribute("width", two ? Math.max(3.5, state.BW * info.bar2) : 0);
  });
}

function update(state, data) {
  const capacity = data.mode === "capacity";
  state.wrap.classList.toggle("mode-capacity", capacity);
  state.busInfo = data.buses;
  state.lineInfo = data.lines;
  state.subInfo = data.subs;
  paint(state, data.layout, data.buses, data.lines, data.subs, capacity);
  mark(state, data.selected, data.label);

  state.legend.replaceChildren();
  data.legend.forEach((item) => {
    const span = document.createElement("span");
    const sw = document.createElement("i");
    if (item.line) { sw.className = "ln-sw"; sw.style.background = item.line; }
    else if (item.tile) { sw.className = "tile-sw"; sw.style.borderColor = item.tile; }
    else if (item.bar) { sw.className = "bar-sw"; }
    else if (item.dots) { sw.className = "dot-sw"; }
    span.appendChild(sw);
    span.appendChild(document.createTextNode(item.label));
    state.legend.appendChild(span);
  });
  if (data.foot) {
    const foot = document.createElement("span");
    foot.className = "foot";
    foot.textContent = data.foot;
    state.legend.appendChild(foot);
  }
}

function tick(state, now) {
  if (!state.wrap.isConnected) { state.running = false; return; }
  const dt = Math.min((now - state.last) / 1000, 0.1);
  state.last = now;
  // Speed eases toward its target over about 0.4 s, so a flow that reverses
  // slows, stops and turns around instead of jumping.
  const ease = Math.min(1, dt / 0.4);
  for (const e of state.edges) {
    e.speed += (e.target - e.speed) * ease;
    if (REDUCED) continue;
    e.phase -= e.speed * dt;
    if (e.phase < -1500 || e.phase > 1500) e.phase %= 15;
    e.dots.setAttribute("stroke-dashoffset", e.phase.toFixed(2));
  }
  requestAnimationFrame((t) => tick(state, t));
}

const FRAME_MS = REDUCED ? 1000 : 650;

function stopPlay(state) {
  if (state.playTimer) clearTimeout(state.playTimer);
  state.playTimer = null;
  state.wrap.classList.remove("playing");
}

// Play one solved grid per hour, with a clock, then return to the live view.
function playDay(state, data) {
  stopPlay(state);
  const frames = data.frames;
  state.wrap.classList.add("playing");
  state.clockWhat.textContent = data.day_label || "";
  let i = 0;
  const step = () => {
    if (!state.wrap.isConnected) return;
    if (i >= frames.length) {
      state.playTimer = setTimeout(() => {
        stopPlay(state);
        paint(state, data.layout, data.buses, data.lines, data.subs, false);
      }, 1800);
      return;
    }
    const f = frames[i];
    paint(state, data.layout, f.buses, f.lines, f.subs, false);
    state.clockTime.textContent = f.clock;
    state.clockSun.style.width = `${Math.round(f.sun * 100)}%`;
    state.clockStatus.textContent = f.status;
    state.clock.classList.toggle("alarm", !!f.alarm);
    i += 1;
    state.playTimer = setTimeout(step, FRAME_MS);
  };
  step();
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  if (!data || !data.layout) return;
  let state = parentElement.__headroomBoard;
  if (!state || state.layoutId !== data.layout.id || !state.wrap.isConnected) {
    state = build(parentElement, data);
    parentElement.__headroomBoard = state;
  }
  state.setTrigger = setTriggerValue;
  update(state, data);
  if (data.frames && data.play && data.play !== state.lastPlay) {
    state.lastPlay = data.play;
    playDay(state, data);
  } else if (!data.frames) {
    stopPlay(state);
  }
  if (!state.running) {
    state.running = true;
    state.last = performance.now();
    requestAnimationFrame((t) => tick(state, t));
  }
}
"""

_board = st.components.v2.component("headroom_board", css=CSS, js=JS)


def show_board(data, key="board"):
    """Draw the board. Returns the bus number clicked on it, or None."""
    result = _board(key=key, data=data, on_selected_change=lambda: None)
    clicked = result.get("selected") if result else None
    return int(clicked) if clicked is not None else None
