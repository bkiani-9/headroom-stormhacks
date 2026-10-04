"""A schematic single-line diagram of the grid, as a custom Streamlit
component (st.components.v2). Inline HTML, CSS and JS only: no external
libraries, so it works offline.

The grid is radial, so each substation's area is drawn as a tidy tree:
the substation as a large tile on the left, distance from it (in lines)
along x, branches stacked along y, buses as uniform rounded squares, lines
as right-angled connectors. Particles move along each line in the real
direction of active power, faster for more power.
"""
import networkx as nx
import pandapower.topology as top
import streamlit as st

from network_plot import colour_for, loading_used, room_left

# Status colours for the Capacity diagram (the same palette as the scale).
FIRM = "#0ca30c"
FLEXIBLE = "#fab219"
CANT = "#d03b3b"
BUS_FILL = "#f4f3ef"


def tree_layout(net):
    """Tidy-tree positions for every 20 kV bus, from the substations out.

    Returns a dict for the component: substations (tiles), buses (depth =
    lines from the substation, row = vertical slot) and edges (parent bus,
    child bus, line). The longest branch of each bus continues straight on
    its row; other branches open new rows below. Lines that close a loop
    (normally-open points) are not part of the tree and are not drawn.
    """
    graph = top.create_nxgraph(net, include_trafos=False, respect_switches=True)
    subs, buses, edges = [], [], []
    row = 0
    for number, root in enumerate(net.trafo.lv_bus):
        tree = nx.bfs_tree(graph, root)
        height = {}  # longest path below each bus, to pick the straight branch
        for bus in reversed(list(nx.topological_sort(tree))):
            height[bus] = 1 + max((height[c] for c in tree.successors(bus)), default=0)

        start = row
        positions = {}

        def place(bus, depth, r):
            positions[bus] = (depth, r)
            children = sorted(tree.successors(bus), key=lambda c: -height[c])
            next_row = r
            for i, child in enumerate(children):
                if i > 0:
                    next_row += 1
                next_row = place(child, depth + 1, next_row)
            return next_row

        last = place(root, 0, row)
        for bus, (depth, r) in positions.items():
            if bus != root:
                buses.append({"id": int(bus), "depth": depth, "row": r})
        for parent, child in tree.edges():
            lines = [k[1] for k in graph.get_edge_data(parent, child) if k[0] == "line"]
            edges.append({"line": int(lines[0]), "parent": int(parent), "child": int(child)})
        subs.append({"bus": int(root), "row": start, "rows": last - start + 1,
                     "label": f"Substation {number + 1}"})
        row = last + 3  # a gap between the two substation areas

    return {
        "id": f"{len(buses)}-{len(edges)}-{row}",
        "rows": row - 2,
        "cols": max(b["depth"] for b in buses) + 1,
        "subs": subs, "buses": buses, "edges": edges,
    }


def _line_flows(net, layout):
    """Colour by loading, and direction (+1 parent to child, -1 back toward
    the substation) and size of active power, for each drawn line."""
    out = []
    for e in layout["edges"]:
        line = e["line"]
        p = float(net.res_line.p_from_mw[line])
        loading = float(net.res_line.loading_percent[line])
        sending = net.line.from_bus[line] if p >= 0 else net.line.to_bus[line]
        out.append({
            "colour": colour_for(room_left(loading_used(loading))),
            "dir": 1 if sending == e["parent"] else -1,
            "mw": abs(p),
            "label": f"Line {line}: {abs(p):.1f} MW, loaded {loading:.0f}%",
        })
    return out


def project_data(net, layout, mw, capacity, selected):
    """Data for the Project check diagram: flows from `net` (the latest power
    flow) and, on each bus, a gauge of how much of that bus's capacity a
    project of `mw` would use. `capacity` maps bus -> MW."""
    buses = []
    for b in layout["buses"]:
        cap = float(capacity.get(b["id"], 0.0))
        used = mw / cap if cap > 0 else (1.0 if mw > 0 else 0.0)
        buses.append({
            "fill": BUS_FILL,
            "gauge": min(used, 1.0),
            "gauge_colour": colour_for(room_left(used * 100)),
            "label": (f"Bus {b['id']}: takes {cap:.1f} MW; "
                      f"{mw:g} MW uses {used:.0%} of it"),
        })
    return {"layout": layout, "mode": "project", "selected": int(selected),
            "lines": _line_flows(net, layout), "buses": buses}


def capacity_data(net, layout, size, firm, best, selected):
    """Data for the Capacity diagram: today's flows, and each bus coloured
    firm (fits in 95% of conditions), flexible only (fits in some) or can't
    (fits in none) at `size` MW. `firm` and `best` map bus -> MW."""
    buses = []
    for b in layout["buses"]:
        f = float(firm.get(b["id"], 0.0))
        top_mw = max(float(best.get(b["id"], 0.0)), f)
        if size <= f + 1e-9:
            fill, kind = FIRM, "firm"
        elif size <= top_mw + 1e-9:
            fill, kind = FLEXIBLE, "flexible only"
        else:
            fill, kind = CANT, "can't"
        buses.append({
            "fill": fill, "gauge": None, "gauge_colour": None,
            "label": (f"Bus {b['id']}: {kind} at {size:g} MW (firm up to "
                      f"{f:.1f} MW, up to {top_mw:.1f} MW at best)"),
        })
    return {"layout": layout, "mode": "capacity", "selected": int(selected),
            "lines": _line_flows(net, layout), "buses": buses}


CSS = """
.hr-wrap { position: relative; width: 100%; font-family: inherit; }
.hr-wrap svg { width: 100%; height: auto; display: block; }
.ln { fill: none; stroke-width: 5; stroke-linecap: round; stroke-linejoin: round;
      transition: stroke 300ms ease; }
.pt { fill: #ffffff; stroke: #0b0b0b; stroke-width: 1; pointer-events: none;
      transition: opacity 300ms ease; }
.bus rect.sq { stroke: #898781; stroke-width: 1.5; cursor: pointer;
               transition: fill 300ms ease, stroke 300ms ease; }
.bus .track { fill: #e1e0d9; pointer-events: none; }
.bus .gauge { pointer-events: none; transition: width 300ms ease, fill 300ms ease; }
.bus.sel rect.sq { stroke: #0b0b0b; stroke-width: 3;
                   filter: drop-shadow(0 0 6px #fab219) drop-shadow(0 0 3px #fab219);
                   animation: hr-glow 1.4s ease-in-out infinite; }
@keyframes hr-glow { 50% { filter: drop-shadow(0 0 10px #fab219) drop-shadow(0 0 4px #fab219); } }
.sub rect { fill: #0b0b0b; cursor: pointer; }
.sub text { fill: #ffffff; font-size: 15px; font-weight: 700; pointer-events: none; }
.hr-tip { position: absolute; pointer-events: none; background: #0b0b0b; color: #ffffff;
          padding: 6px 10px; border-radius: 6px; font-size: 18px; white-space: nowrap;
          opacity: 0; transition: opacity 150ms ease; z-index: 2; }
"""

JS = r"""
const CW = 34, RH = 22, S = 14, TILE_W = 120, PAD = 10;
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

function build(host, data) {
  host.querySelectorAll(".hr-wrap").forEach((n) => n.remove());
  const L = data.layout;
  const wrap = document.createElement("div");
  wrap.className = "hr-wrap";
  host.appendChild(wrap);
  const width = TILE_W + 30 + L.cols * CW + PAD;
  const height = L.rows * RH + 2 * PAD;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}` }, wrap);
  const tip = document.createElement("div");
  tip.className = "hr-tip";
  wrap.appendChild(tip);

  const gLines = el("g", {}, svg), gPts = el("g", {}, svg);
  const gSubs = el("g", {}, svg), gBuses = el("g", {}, svg);
  const pos = {};
  const xOf = (depth) => TILE_W + 30 + (depth - 1) * CW;
  const yOf = (row) => PAD + row * RH + RH / 2;
  L.subs.forEach((s) => { pos[s.bus] = { x: TILE_W, y: yOf(s.row), sub: true }; });
  L.buses.forEach((b) => { pos[b.id] = { x: xOf(b.depth), y: yOf(b.row) }; });

  const state = { layoutId: L.id, wrap, svg, tip, buses: {}, edges: [], selected: null };

  // Show a label near the pointer.
  const showTip = (evt, text) => {
    const box = wrap.getBoundingClientRect();
    tip.textContent = text;
    tip.style.left = `${evt.clientX - box.left + 14}px`;
    tip.style.top = `${evt.clientY - box.top + 14}px`;
    tip.style.opacity = 1;
  };
  const hideTip = () => { tip.style.opacity = 0; };

  // Lines: right-angled connectors from the parent's right side to the child's left side.
  L.edges.forEach((e, i) => {
    const a = pos[e.parent], b = pos[e.child];
    const x0 = a.sub ? a.x : a.x + S / 2, y0 = a.y;
    const x1 = b.x - S / 2, y1 = b.y;
    const xm = Math.max(x0, x1 - CW * 0.4);
    const pts = y0 === y1 ? [[x0, y0], [x1, y1]] : [[x0, y0], [xm, y0], [xm, y1], [x1, y1]];
    const d = "M" + pts.map((p) => p.join(",")).join(" L");
    const path = el("path", { d, class: "ln", stroke: "#c3c2b7" }, gLines);
    const segs = [];
    let total = 0;
    for (let k = 1; k < pts.length; k++) {
      const len = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
      segs.push({ from: pts[k - 1], to: pts[k], start: total, len });
      total += len;
    }
    const n = Math.max(1, Math.floor(total / 26));
    const dots = [];
    for (let k = 0; k < n; k++) dots.push(el("circle", { r: 2.6, class: "pt", opacity: 0 }, gPts));
    path.addEventListener("mousemove", (evt) => showTip(evt, state.lineLabels[i]));
    path.addEventListener("mouseleave", hideTip);
    state.edges.push({ path, segs, total, dots, phase: Math.random() * total,
                       speed: 0, target: 0, opacity: 0, parent: e.parent, child: e.child });
  });

  // Substations: large rounded tiles on the left; clicking one picks its 20 kV bus.
  L.subs.forEach((s) => {
    const h = Math.max(RH * 2, RH * 3);
    const g = el("g", { class: "sub" }, gSubs);
    el("rect", { x: 0, y: yOf(s.row) - h / 2, width: TILE_W, height: h, rx: 12 }, g);
    const t = el("text", { x: TILE_W / 2, y: yOf(s.row) + 5, "text-anchor": "middle" }, g);
    t.textContent = s.label;
    g.addEventListener("click", () => state.setTrigger("selected", s.bus));
    g.addEventListener("mousemove", (evt) => showTip(evt, `${s.label} (bus ${s.bus})`));
    g.addEventListener("mouseleave", hideTip);
  });

  // Buses: uniform rounded squares, each with a small gauge along its bottom.
  L.buses.forEach((b, i) => {
    const p = pos[b.id];
    const g = el("g", { class: "bus" }, gBuses);
    const sq = el("rect", { class: "sq", x: p.x - S / 2, y: p.y - S / 2, width: S,
                            height: S, rx: 3.5, fill: "#f4f3ef" }, g);
    const track = el("rect", { class: "track", x: p.x - S / 2 + 2, y: p.y + S / 2 - 5,
                               width: S - 4, height: 3, rx: 1 }, g);
    const gauge = el("rect", { class: "gauge", x: p.x - S / 2 + 2, y: p.y + S / 2 - 5,
                               height: 3, rx: 1 }, g);
    gauge.style.width = "0px";
    g.addEventListener("click", () => state.setTrigger("selected", b.id));
    g.addEventListener("mousemove", (evt) => showTip(evt, state.busLabels[i]));
    g.addEventListener("mouseleave", hideTip);
    state.buses[b.id] = { g, sq, track, gauge, i };
  });
  return state;
}

function update(state, data) {
  state.lineLabels = data.lines.map((l) => l.label);
  state.busLabels = data.buses.map((b) => b.label);
  data.lines.forEach((l, i) => {
    const e = state.edges[i];
    e.path.setAttribute("stroke", l.colour);
    // Particle speed in drawing units per second: faster for more power.
    const speed = l.mw < 0.01 ? 0 : 14 + 70 * Math.min(l.mw / 8, 1);
    e.target = l.dir * speed;
    e.opacity = l.mw < 0.01 ? 0 : 1;
  });
  data.layout.buses.forEach((b, i) => {
    const info = data.buses[i], node = state.buses[b.id];
    node.sq.setAttribute("fill", info.fill);
    const showGauge = info.gauge !== null;
    node.track.style.display = showGauge ? "" : "none";
    node.gauge.style.display = showGauge ? "" : "none";
    if (showGauge) {
      node.gauge.style.width = `${(S - 4) * info.gauge}px`;
      node.gauge.setAttribute("fill", info.gauge_colour);
    }
    node.g.classList.toggle("sel", b.id === data.selected);
  });
}

function pointAt(e, s) {
  s = ((s % e.total) + e.total) % e.total;
  for (const seg of e.segs) {
    if (s <= seg.start + seg.len) {
      const f = seg.len ? (s - seg.start) / seg.len : 0;
      return [seg.from[0] + (seg.to[0] - seg.from[0]) * f,
              seg.from[1] + (seg.to[1] - seg.from[1]) * f];
    }
  }
  const last = e.segs[e.segs.length - 1];
  return last.to;
}

function tick(state, now) {
  if (!state.wrap.isConnected) { state.running = false; return; }
  const dt = Math.min((now - state.last) / 1000, 0.1);
  state.last = now;
  // Ease speed and direction toward their targets over about 300 ms, so a
  // reversal slows, stops and turns around instead of jumping.
  const ease = Math.min(1, dt / 0.3);
  for (const e of state.edges) {
    e.speed += (e.target - e.speed) * ease;
    e.phase += e.speed * dt;
    const n = e.dots.length;
    for (let k = 0; k < n; k++) {
      const [x, y] = pointAt(e, e.phase + (k * e.total) / n);
      e.dots[k].setAttribute("cx", x);
      e.dots[k].setAttribute("cy", y);
      e.dots[k].setAttribute("opacity", e.opacity);
    }
  }
  requestAnimationFrame((t) => tick(state, t));
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  if (!data || !data.layout) return;
  let state = parentElement.__headroom;
  if (!state || state.layoutId !== data.layout.id || !state.wrap.isConnected) {
    state = build(parentElement, data);
    parentElement.__headroom = state;
  }
  state.setTrigger = setTriggerValue;
  update(state, data);
  if (!state.running) {
    state.running = true;
    state.last = performance.now();
    requestAnimationFrame((t) => tick(state, t));
  }
}
"""

_component = st.components.v2.component("headroom_diagram", css=CSS, js=JS)


def show_diagram(data, key):
    """Draw the diagram. Returns the bus number clicked in it, or None."""
    result = _component(key=key, data=data, on_selected_change=lambda: None)
    clicked = result.get("selected") if result else None
    return int(clicked) if clicked is not None else None
