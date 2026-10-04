"""Electrical space: the grid drawn in the hyperbolic plane, for app_v2.py.

On a map, distance is kilometres. To a grid, the distance that matters is
electrical: the impedance (ohms) of the lines between two buses. Voltage
rise grows with it (dV/V is about P x R / V^2), which is why the far end
of a feeder runs out of room first.

A radial grid is a tree, and trees fit the hyperbolic plane far better
than a flat page: room grows exponentially with distance from any point,
so every branch keeps its own space. Here every line becomes a geodesic of
hyperbolic length K x its impedance, laid out from the 110 kV grid outward
(each bus's own frame: a feeder continues straight, a side branch turns
off at BRANCH_ANGLE). The view is the Poincare disk; moving the centre to
another bus is a hyperbolic translation (a Mobius map), which keeps every
distance and angle.

measure_fit() checks how well each drawing keeps the real electrical
distances between all pairs of buses: this one against the best flat
drawing (classical multidimensional scaling).
"""
import cmath
import math

import networkx as nx
import numpy as np
import pandapower.topology as top
import streamlit as st

from control_room import _flows

K = 1.6  # hyperbolic length per ohm
BRANCH_ANGLE = math.radians(80)
ROOT = "grid"
RINGS_OHM = [0.25, 0.5, 1, 2]


def _move_to_origin(a, z):
    return (z - a) / (1 - a.conjugate() * z)


def _move_from_origin(a, w):
    return (w + a) / (1 + a.conjugate() * w)


def _tree(net, layout):
    """Children, impedance of the line into each node (ohms), and the
    order to visit nodes from the 110 kV grid outward."""
    graph = top.create_nxgraph(net, calc_branch_impedances=True, include_trafos=False,
                               respect_switches=True)
    children, z_in, line_of = {ROOT: []}, {}, {}
    for s in layout["subs"]:
        trafo = net.trafo[net.trafo.lv_bus == s["bus"]].iloc[0]
        # Transformer impedance seen from the 20 kV side: vk x V^2 / S.
        z_in[s["bus"]] = float(trafo.vk_percent / 100 * trafo.vn_lv_kv ** 2 / trafo.sn_mva)
        children[ROOT].append(s["bus"])
    for e in layout["edges"]:
        children.setdefault(e["parent"], []).append(e["child"])
        z_in[e["child"]] = float(graph.get_edge_data(e["parent"], e["child"])[("line", e["line"])]["z_ohm"])
        line_of[e["child"]] = e["line"]
    order = [ROOT]
    for n in order:
        order.extend(children.get(n, []))
    size = {}
    for n in reversed(order):
        size[n] = 1 + sum(size[c] for c in children.get(n, []))
    return children, z_in, order, size


def embed(net, layout):
    """Position (a point in the unit disk) of the grid and every bus."""
    children, z_in, order, size = _tree(net, layout)
    parent = {c: n for n in order for c in children.get(n, [])}
    pos = {ROOT: 0j}
    for i, s in enumerate(children[ROOT]):
        pos[s] = math.tanh(K * z_in[s] / 2) * cmath.exp(1j * (math.pi / 2 + i * math.pi))
    for n in order[1:]:
        kids = sorted(children.get(n, []), key=lambda c: (-size[c], c))
        if not kids:
            continue
        p = pos[n]
        back = _move_to_origin(p, pos[parent[n]])
        away = cmath.phase(-back)
        if len(kids) == 1:
            angles = [away]
        elif parent[n] == ROOT:  # a substation: its two feeders and the 110 kV side, 120 degrees apart
            angles = [away - math.radians(60), away + math.radians(60)]
        else:  # the feeder goes straight on; the branch turns off
            angles = [away, away + (BRANCH_ANGLE if n % 2 else -BRANCH_ANGLE)]
        for c, phi in zip(kids, angles):
            pos[c] = _move_from_origin(p, math.tanh(K * z_in[c] / 2) * cmath.exp(1j * phi))
    return pos, children, z_in, order


def measure_fit(net, layout):
    """Mean relative error of all pair distances (ohms): this hyperbolic
    drawing, and the best flat drawing (classical MDS)."""
    pos, children, z_in, order = embed(net, layout)
    tree = nx.Graph()
    for n in order:
        for c in children.get(n, []):
            tree.add_edge(n, c, w=z_in[c])
    nodes = order[1:]
    dist = dict(nx.all_pairs_dijkstra_path_length(tree, weight="w"))
    true = np.array([[dist[a][b] for b in nodes] for a in nodes])
    upper = np.triu_indices(len(nodes), 1)

    def hyperbolic(a, b):
        ratio = abs(a - b) / abs(1 - a.conjugate() * b)
        return 2 * math.atanh(min(ratio, 1 - 1e-15)) / K

    curved = np.array([[hyperbolic(pos[a], pos[b]) for b in nodes] for a in nodes])
    n = len(nodes)
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ (true ** 2) @ centring
    values, vectors = np.linalg.eigh(gram)
    top2 = np.argsort(values)[::-1][:2]
    flat_xy = vectors[:, top2] * np.sqrt(np.maximum(values[top2], 0))
    flat = np.sqrt(((flat_xy[:, None, :] - flat_xy[None, :, :]) ** 2).sum(-1))
    t = true[upper]
    return (float(np.mean(np.abs(curved[upper] - t) / t)),
            float(np.mean(np.abs(flat[upper] - t) / t)))


def disk_layout(net, layout):
    """Everything the disk draws that doesn't change: nodes and lines."""
    pos, children, z_in, order = embed(net, layout)
    sub_label = {s["bus"]: s["label"] for s in layout["subs"]}
    z_sub, hops, home = {}, {}, {}
    for s in children[ROOT]:
        z_sub[s], hops[s], home[s] = 0.0, 0, sub_label[s]
    for n in order[1:]:
        for c in children.get(n, []):
            z_sub[c] = z_sub[n] + z_in[c]
            hops[c] = hops[n] + 1
            home[c] = home[n]
    nodes = [{"id": ROOT, "x": 0.0, "y": 0.0, "kind": "root", "label": "110 kV grid"}]
    for n in order[1:]:
        kind = "sub" if n in sub_label else "bus"
        nodes.append({"id": int(n), "x": round(pos[n].real, 9), "y": round(pos[n].imag, 9),
                      "kind": kind,
                      "label": sub_label[n] if kind == "sub" else f"Bus {n}",
                      "detail": (f"{z_sub[n]:.2f} ohms from {home[n]}, {hops[n]} lines out"
                                 if kind == "bus" else "110/20 kV substation")})
    edges = [{"a": ROOT, "b": int(s), "kind": "trafo"} for s in children[ROOT]]
    edges += [{"a": int(e["parent"]), "b": int(e["child"]), "kind": "line"} for e in layout["edges"]]
    return {"id": f"disk-{layout['id']}-{K}", "k": K, "nodes": nodes, "edges": edges,
            "rings": [{"ohm": r, "radius": round(math.tanh(K * r / 2), 6)} for r in RINGS_OHM]}


def disk_data(net, layout, disk, bus_states, selected):
    """The disk's data for one grid state: `bus_states` is the board's
    per-bus list (bar, colour, state), in layout["buses"] order. The view
    opens centred on the substation of the `selected` bus."""
    row = next((b["row"] for b in layout["buses"] if b["id"] == selected), None)
    start = next((s["bus"] for s in layout["subs"]
                  if row is not None and s["top"] <= row <= s["bottom"]), None)
    if selected in [s["bus"] for s in layout["subs"]]:
        start = selected
    states = {b["id"]: {"state": s["state"], "colour": s["colour"]}
              for b, s in zip(layout["buses"], bus_states)}
    flows = []
    for s in layout["subs"]:
        trafo = int(net.trafo.index[net.trafo.lv_bus == s["bus"]][0])
        p = float(net.res_trafo.p_hv_mw[trafo])  # + means power comes down from 110 kV
        flows.append({"dir": 1 if p >= 0 else -1, "mw": round(abs(p), 3), "stroke": "#8C8C8C"})
    flows += [{k: f[k] for k in ("dir", "mw", "stroke")} for f in _flows(net, layout)]
    return {"disk": disk, "states": {str(k): v for k, v in states.items()},
            "flows": flows, "selected": int(selected),
            "start": int(start) if start is not None else ROOT}


CSS = """
:host { display: block; }
.hd { position: relative; color: #FFFFFF; user-select: none; max-width: 660px; margin: 0 auto;
      font-family: "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif; }
.hd svg { display: block; width: 100%; height: auto; }
.rim { fill: #0E0E0E; stroke: #4A4A4A; stroke-width: 1.5; }
.ring { fill: none; stroke: #2A2A2A; stroke-width: 1; stroke-dasharray: 3 6; }
.ring-label { fill: #7A7A7A; font-size: 13px; text-anchor: start; }
.geo { fill: none; stroke-linecap: round; }
.dots { fill: none; stroke: #FFFFFF; stroke-linecap: round; stroke-dasharray: 0 13;
        pointer-events: none; }
.node { cursor: pointer; }
.node rect { fill: #161616; stroke: #A0A0A0; }
.node:hover rect { stroke: #FFFFFF; }
.node.warn rect { stroke: #FFA61A; fill: #2E1E05; }
.node.over rect { stroke: #FF4D57; fill: #33100F; filter: drop-shadow(0 0 4px rgba(255,77,87,.8)); }
.node.sub rect { fill: #161616; stroke: #C8C8C8; }
.node.root circle { fill: #161616; stroke: #C8C8C8; stroke-width: 2; }
.node.focus rect, .node.focus circle { stroke: #FFFFFF; stroke-width: 3; }
.node text { fill: #FFFFFF; font-size: 14px; text-anchor: middle; pointer-events: none; }
.hud { position: absolute; top: 0; left: 0; max-width: 44%; pointer-events: none; }
.hud .t { font-size: 22px; font-weight: 600; line-height: 1.2; }
.hud .d { font-size: 14px; color: #ADADAD; margin-top: 3px; line-height: 1.35; }
.hud button { pointer-events: auto; margin-top: 10px; background: transparent; color: #FFFFFF;
              border: 1.5px solid #A0A0A0; border-radius: 10px; padding: 5px 12px;
              font: inherit; font-size: 14px; cursor: pointer; }
.hud button:hover { border-color: #FFFFFF; }
.tipd { position: absolute; pointer-events: none; opacity: 0; z-index: 3;
        background: rgba(22,22,22,.97); border: 1px solid #6E6E6E; border-radius: 9px;
        padding: 6px 10px; font-size: 13.5px; transition: opacity .12s ease; }
.tipd b { display: block; font-size: 14.5px; }
.tipd span { color: #D2D2D2; }
"""

JS = r"""
const NS = "http://www.w3.org/2000/svg";
const C = 500, R = 480;
const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const cx = (re, im) => ({ re, im });
const add = (a, b) => cx(a.re + b.re, a.im + b.im);
const sub = (a, b) => cx(a.re - b.re, a.im - b.im);
const mul = (a, b) => cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const conj = (a) => cx(a.re, -a.im);
const abs = (a) => Math.hypot(a.re, a.im);
const div = (a, b) => { const d = b.re * b.re + b.im * b.im;
  return cx((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d); };
const ONE = cx(1, 0);
// Hyperbolic translation taking `a` to the centre, and its inverse.
const toCentre = (a, z) => div(sub(z, a), sub(ONE, mul(conj(a), z)));
const fromCentre = (a, w) => div(add(w, a), add(ONE, mul(conj(a), w)));
const scr = (z) => [C + R * z.re, C - R * z.im];

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

// The geodesic between two points of the disk: an arc of the circle through
// them that meets the rim at right angles (a straight line through the centre).
function geodesic(p, q) {
  const P = scr(p), Q = scr(q);
  const straight = `M ${P[0].toFixed(2)} ${P[1].toFixed(2)} L ${Q[0].toFixed(2)} ${Q[1].toFixed(2)}`;
  if (Math.abs(p.re * q.im - p.im * q.re) < 1e-9) return straight;
  const o = abs(p) >= abs(q) ? p : q;
  const r2 = o.re * o.re + o.im * o.im;
  if (r2 < 1e-12) return straight;
  const inv = cx(o.re / r2, o.im / r2);
  const ax = p.re, ay = p.im, bx = q.re, by = q.im, ccx = inv.re, ccy = inv.im;
  const d = 2 * (ax * (by - ccy) + bx * (ccy - ay) + ccx * (ay - by));
  if (Math.abs(d) < 1e-12) return straight;
  const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = ccx * ccx + ccy * ccy;
  const ux = (a2 * (by - ccy) + b2 * (ccy - ay) + c2 * (ay - by)) / d;
  const uy = (a2 * (ccx - bx) + b2 * (ax - ccx) + c2 * (bx - ax)) / d;
  const rad = Math.hypot(ax - ux, ay - uy) * R;
  if (!isFinite(rad) || rad > 1e6) return straight;
  const U = scr(cx(ux, uy));
  const cross = (P[0] - U[0]) * (Q[1] - U[1]) - (P[1] - U[1]) * (Q[0] - U[0]);
  return `M ${P[0].toFixed(2)} ${P[1].toFixed(2)} A ${rad.toFixed(2)} ${rad.toFixed(2)} 0 0 ` +
         `${cross > 0 ? 1 : 0} ${Q[0].toFixed(2)} ${Q[1].toFixed(2)}`;
}

function build(host, data) {
  host.querySelectorAll(".hd").forEach((n) => n.remove());
  const D = data.disk;
  const wrap = document.createElement("div");
  wrap.className = "hd";
  host.appendChild(wrap);
  const svg = el("svg", { viewBox: "0 0 1000 1000", role: "img",
                          "aria-label": "The grid in the hyperbolic plane" }, wrap);
  el("circle", { cx: C, cy: C, r: R, class: "rim" }, svg);
  const gRings = el("g", {}, svg);
  D.rings.forEach((ring) => {
    el("circle", { cx: C, cy: C, r: ring.radius * R, class: "ring" }, gRings);
    const t = el("text", { x: C + ring.radius * R + 6, y: C - 6, class: "ring-label" }, gRings);
    t.textContent = `${ring.ohm} Ω`;
  });
  const gEdges = el("g", {}, svg), gDots = el("g", {}, svg), gNodes = el("g", {}, svg);

  const hud = document.createElement("div");
  hud.className = "hud";
  hud.innerHTML = '<div class="t"></div><div class="d"></div><button type="button">Back to the 110 kV grid</button>';
  wrap.appendChild(hud);
  const tip = document.createElement("div");
  tip.className = "tipd";
  wrap.appendChild(tip);

  const state = { id: D.id, wrap, svg, hud, tip, nodes: {}, edges: [], focus: cx(0, 0),
                  focusId: "grid", anim: null };
  const showTip = (evt, title, detail) => {
    const box = wrap.getBoundingClientRect();
    tip.replaceChildren();
    const b = document.createElement("b"); b.textContent = title; tip.appendChild(b);
    const s = document.createElement("span"); s.textContent = detail || ""; tip.appendChild(s);
    tip.style.opacity = 1;
    let x = evt.clientX - box.left + 14, y = evt.clientY - box.top + 14;
    if (x + tip.offsetWidth > box.width - 4) x = evt.clientX - box.left - tip.offsetWidth - 12;
    tip.style.left = `${x}px`; tip.style.top = `${y}px`;
  };
  const hideTip = () => { tip.style.opacity = 0; };

  D.nodes.forEach((n) => {
    const g = el("g", { class: `node ${n.kind}`, "data-id": n.id }, gNodes);
    const shape = n.kind === "root" ? el("circle", { r: 14 }, g) : el("rect", { rx: 4 }, g);
    const label = el("text", {}, g);
    const node = { n, g, shape, label, base: cx(n.x, n.y), pos: cx(n.x, n.y) };
    state.nodes[String(n.id)] = node;
    g.addEventListener("click", () => { hideTip(); travel(state, n.id); });
    g.addEventListener("mousemove", (evt) => showTip(evt, n.label, n.detail));
    g.addEventListener("mouseleave", hideTip);
  });
  D.edges.forEach((e) => {
    const path = el("path", { class: "geo" }, gEdges);
    const dots = el("path", { class: "dots" }, gDots);
    state.edges.push({ e, path, dots, phase: Math.random() * 13, speed: 0, target: 0 });
  });
  hud.querySelector("button").addEventListener("click", () => travel(state, "grid"));
  return state;
}

// Redraw every node and line for the current centre.
function layoutAll(state) {
  for (const k in state.nodes) {
    const node = state.nodes[k];
    node.pos = toCentre(state.focus, node.base);
    const r = abs(node.pos);
    const scale = Math.max(0, 1 - r * r);  // how big things look at this depth
    const P = scr(node.pos);
    const kind = node.n.kind;
    const s = (kind === "sub" ? 38 : 18) * scale;
    if (kind === "root") {
      node.shape.setAttribute("cx", P[0]); node.shape.setAttribute("cy", P[1]);
      node.shape.setAttribute("r", Math.max(2, 16 * scale));
    } else {
      node.shape.setAttribute("x", P[0] - s / 2); node.shape.setAttribute("y", P[1] - s / 2);
      node.shape.setAttribute("width", s); node.shape.setAttribute("height", s);
      node.shape.setAttribute("rx", Math.max(1, s * 0.22));
      node.shape.setAttribute("stroke-width", Math.max(0.6, 2 * scale));
    }
    node.g.style.display = scale < 0.004 ? "none" : "";
    const showLabel = kind !== "bus" ? scale > 0.35 : String(node.n.id) === String(state.focusId);
    node.label.textContent = showLabel ? node.n.label : "";
    node.label.setAttribute("x", P[0]);
    node.label.setAttribute("y", P[1] + (kind === "root" ? 16 : s / 2) * Math.max(scale, 0.3) + 16);
  }
  state.edges.forEach((item) => {
    const a = state.nodes[String(item.e.a)], b = state.nodes[String(item.e.b)];
    const d = geodesic(a.pos, b.pos);
    item.path.setAttribute("d", d);
    item.dots.setAttribute("d", d);
    const mid = cx((a.pos.re + b.pos.re) / 2, (a.pos.im + b.pos.im) / 2);
    const scale = Math.max(0, 1 - abs(mid) ** 2);
    item.path.setAttribute("stroke-width", Math.max(0.5, 5 * scale));
    item.dots.setAttribute("stroke-width", Math.max(0, 6 * scale));
    item.dots.style.display = scale < 0.05 ? "none" : "";
  });
  for (const k in state.nodes) {
    state.nodes[k].g.classList.toggle("focus", String(state.nodes[k].n.id) === String(state.focusId));
  }
  const f = state.nodes[String(state.focusId)];
  state.hud.querySelector(".t").textContent = f ? `Centre: ${f.n.label}` : "";
  state.hud.querySelector(".d").textContent = f && f.n.detail ? f.n.detail :
    "Click any bus to travel to it. Rings show electrical distance from the centre.";
}

// Move the centre to another node along the hyperbolic straight line.
function travel(state, id) {
  const target = state.nodes[String(id)];
  if (!target) return;
  const start = state.focus;
  const w = toCentre(start, target.base);
  const len = abs(w);
  state.focusId = id;
  if (len < 1e-9) { layoutAll(state); return; }
  const dist = 2 * Math.atanh(Math.min(len, 1 - 1e-12));
  const t0 = performance.now(), ms = REDUCED ? 1 : 1100;
  const step = (now) => {
    const t = Math.min(1, (now - t0) / ms);
    const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    const r = Math.tanh((e * dist) / 2);
    state.focus = fromCentre(start, cx((w.re / len) * r, (w.im / len) * r));
    layoutAll(state);
    if (t < 1) state.anim = requestAnimationFrame(step);
  };
  if (state.anim) cancelAnimationFrame(state.anim);
  state.anim = requestAnimationFrame(step);
}

function update(state, data) {
  for (const k in state.nodes) {
    const node = state.nodes[k];
    const st = data.states[k];
    node.g.classList.toggle("warn", !!st && st.state === "warn");
    node.g.classList.toggle("over", !!st && st.state === "over");
  }
  data.flows.forEach((f, i) => {
    const item = state.edges[i];
    item.path.style.stroke = f.stroke;
    const moving = f.mw >= 0.02;
    item.target = moving ? f.dir * (8 + 22 * Math.sqrt(f.mw)) : 0;
    item.dots.style.opacity = moving ? 0.9 : 0;
  });
}

function tick(state, now) {
  if (!state.wrap.isConnected) { state.running = false; return; }
  const dt = Math.min((now - state.last) / 1000, 0.1);
  state.last = now;
  const ease = Math.min(1, dt / 0.4);
  for (const item of state.edges) {
    item.speed += (item.target - item.speed) * ease;
    if (REDUCED) continue;
    item.phase -= item.speed * dt;
    if (item.phase < -1300 || item.phase > 1300) item.phase %= 13;
    item.dots.setAttribute("stroke-dashoffset", item.phase.toFixed(2));
  }
  requestAnimationFrame((t) => tick(state, t));
}

export default function (component) {
  const { data, parentElement } = component;
  if (!data || !data.disk) return;
  let state = parentElement.__headroomDisk;
  if (!state || state.id !== data.disk.id || !state.wrap.isConnected) {
    state = build(parentElement, data);
    parentElement.__headroomDisk = state;
    const start = state.nodes[String(data.start)];
    if (start) { state.focus = start.base; state.focusId = data.start; }
    layoutAll(state);
  }
  update(state, data);
  if (!state.running) {
    state.running = true;
    state.last = performance.now();
    requestAnimationFrame((t) => tick(state, t));
  }
}
"""

_disk = st.components.v2.component("headroom_disk", css=CSS, js=JS)


def show_disk(data, key="disk"):
    """Draw the grid in the hyperbolic plane."""
    _disk(key=key, data=data)
