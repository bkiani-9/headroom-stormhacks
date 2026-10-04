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
another bus (by dragging, or clicking a bus) is a hyperbolic isometry (a
Mobius map), which keeps every distance and angle.

Behind the grid is the regular {7,3} tiling of the hyperbolic plane:
identical seven-sided tiles, three at each corner, which can't fit
together on a flat page. It moves with the grid, so the space itself can
be seen moving. The Flat button morphs to flat_positions(): the same grid
on a flat page, fitted to the same distances.

measure_fit() checks how well each drawing keeps the real electrical
distances between all pairs of buses: this one against a flat drawing
fitted to those distances (metric multidimensional scaling, SMACOF).
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


def _smacof(true, start, rounds=1500):
    """Metric MDS by stress majorization (SMACOF): move flat points to
    match the distances in `true` as closely as possible, from `start`."""
    x = start.copy()
    n = len(x)
    for _ in range(rounds):
        gaps = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
        ratio = np.where(gaps > 1e-12, true / np.maximum(gaps, 1e-12), 0.0)
        b = -ratio
        np.fill_diagonal(b, 0.0)
        np.fill_diagonal(b, -b.sum(axis=1))
        x = b @ x / n
    return x


def measure_fit(net, layout):
    """Mean relative error of all pair distances (ohms): this hyperbolic
    drawing, and a flat drawing fitted to the same distances (classical MDS
    as the start, then SMACOF)."""
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
    flat_xy = _smacof(true, vectors[:, top2] * np.sqrt(np.maximum(values[top2], 0)))
    flat = np.sqrt(((flat_xy[:, None, :] - flat_xy[None, :, :]) ** 2).sum(-1))
    t = true[upper]
    return (float(np.mean(np.abs(curved[upper] - t) / t)),
            float(np.mean(np.abs(flat[upper] - t) / t)))


def flat_positions(net, layout):
    """The flat drawing for the morph: every node, the 110 kV grid included,
    placed on a flat page to match the electrical distances (classical MDS,
    then SMACOF), scaled to fit inside the disk."""
    pos, children, z_in, order = embed(net, layout)
    tree = nx.Graph()
    for n in order:
        for c in children.get(n, []):
            tree.add_edge(n, c, w=z_in[c])
    dist = dict(nx.all_pairs_dijkstra_path_length(tree, weight="w"))
    true = np.array([[dist[a][b] for b in order] for a in order])
    n = len(order)
    centring = np.eye(n) - np.ones((n, n)) / n
    values, vectors = np.linalg.eigh(-0.5 * centring @ (true ** 2) @ centring)
    top2 = np.argsort(values)[::-1][:2]
    xy = _smacof(true, vectors[:, top2] * np.sqrt(np.maximum(values[top2], 0)), rounds=800)
    xy = xy - xy.mean(axis=0)
    xy = xy * 0.88 / np.sqrt((xy ** 2).sum(axis=1)).max()
    return {node: (float(x), float(y)) for node, (x, y) in zip(order, xy)}


def disk_layout(net, layout):
    """Everything the disk draws that doesn't change: nodes and lines."""
    pos, children, z_in, order = embed(net, layout)
    flat = flat_positions(net, layout)
    sub_label = {s["bus"]: s["label"] for s in layout["subs"]}
    z_sub, hops, home = {}, {}, {}
    for s in children[ROOT]:
        z_sub[s], hops[s], home[s] = 0.0, 0, sub_label[s]
    for n in order[1:]:
        for c in children.get(n, []):
            z_sub[c] = z_sub[n] + z_in[c]
            hops[c] = hops[n] + 1
            home[c] = home[n]
    nodes = [{"id": ROOT, "x": 0.0, "y": 0.0, "fx": round(flat[ROOT][0], 6),
              "fy": round(flat[ROOT][1], 6), "kind": "root", "label": "110 kV grid",
              "detail": "Where both substations connect"}]
    for n in order[1:]:
        kind = "sub" if n in sub_label else "bus"
        nodes.append({"id": int(n), "x": round(pos[n].real, 12), "y": round(pos[n].imag, 12),
                      "fx": round(flat[n][0], 6), "fy": round(flat[n][1], 6),
                      "kind": kind,
                      "label": sub_label[n] if kind == "sub" else f"Bus {n}",
                      "detail": (f"{z_sub[n]:.2f} ohms from {home[n]}, {hops[n]} lines out"
                                 if kind == "bus" else "110/20 kV substation")})
    edges = [{"a": ROOT, "b": int(s), "kind": "trafo"} for s in children[ROOT]]
    edges += [{"a": int(e["parent"]), "b": int(e["child"]), "kind": "line"} for e in layout["edges"]]
    return {"id": f"disk-{layout['id']}-{K}", "k": K, "nodes": nodes, "edges": edges,
            "rings": [{"ohm": r, "radius": round(math.tanh(K * r / 2), 6)} for r in RINGS_OHM]}


def disk_data(net, layout, disk, bus_states, selected, fit=None):
    """The disk's data for one grid state: `bus_states` is the board's
    per-bus list (bar, colour, state), in layout["buses"] order. The view
    opens centred on the substation of the `selected` bus. `fit` is
    measure_fit()'s (hyperbolic error, flat error), for the flat view's note."""
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
    curved, flat = fit if fit else (None, None)
    return {"disk": disk, "states": {str(k): v for k, v in states.items()},
            "flows": flows, "selected": int(selected),
            "start": int(start) if start is not None else ROOT,
            "fit": {"curved": curved, "flat": flat}}


CSS = """
:host { display: block; }
.hd { position: relative; color: #FFFFFF; user-select: none; max-width: 660px; margin: 0 auto;
      font-family: "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif; }
.hd svg { display: block; width: 100%; height: auto; touch-action: none; cursor: grab; }
.hd svg.dragging { cursor: grabbing; }
.rim { fill: #0E0E0E; stroke: #4A4A4A; stroke-width: 1.5; }
.tiles { fill: none; stroke: #2F2F2F; stroke-width: 1.2; pointer-events: none; }
.ring { fill: none; stroke: #343434; stroke-width: 1; stroke-dasharray: 3 6; pointer-events: none; }
.ring-label { fill: #8A8A8A; font-size: 13px; text-anchor: start; pointer-events: none; }
.geo { fill: none; stroke-linecap: round; pointer-events: none; }
.dots { fill: none; stroke: #FFFFFF; stroke-linecap: round; stroke-dasharray: 0 13;
        pointer-events: none; }
.node { cursor: pointer; }
.hd .node circle.hit { fill: transparent; stroke: none; }
.node rect { fill: #161616; stroke: #A0A0A0; }
.node:hover rect { stroke: #FFFFFF; }
.node.warn rect { stroke: #FFA61A; fill: #2E1E05; }
.node.over rect { stroke: #FF4D57; fill: #33100F; filter: drop-shadow(0 0 4px rgba(255,77,87,.8)); }
.node.sub rect { fill: #161616; stroke: #C8C8C8; }
.node.root circle { fill: #161616; stroke: #C8C8C8; stroke-width: 2; }
.node.focus rect, .node.focus circle { stroke: #FFFFFF; stroke-width: 3; }
.node text { fill: #FFFFFF; font-size: 14px; text-anchor: middle; pointer-events: none;
             paint-order: stroke; stroke: #0E0E0E; stroke-width: 4px; }
.hud { position: absolute; top: 0; left: 0; max-width: 50%; pointer-events: none;
       text-shadow: 0 0 3px #0A0A0A, 0 0 6px #0A0A0A; }
.hud .t { font-size: 22px; font-weight: 600; line-height: 1.2; }
.hud .d { font-size: 14px; color: #ADADAD; margin-top: 3px; line-height: 1.35; }
.bar button, .modes button { background: #0A0A0A; color: #FFFFFF;
              border: 1.5px solid #A0A0A0; border-radius: 10px; padding: 5px 12px;
              font: inherit; font-size: 14px; cursor: pointer; white-space: nowrap; }
.bar button:hover, .modes button:hover { border-color: #FFFFFF; }
.bar button:focus-visible, .modes button:focus-visible { outline: 2px solid #FFFFFF; outline-offset: 2px; }
.modes { position: absolute; top: 0; right: 0; display: flex; gap: 0; }
.modes button { border-radius: 0; margin-left: -1.5px; }
.modes button:first-child { border-radius: 10px 0 0 10px; }
.modes button:last-child { border-radius: 0 10px 10px 0; }
.modes button.on { background: #FFFFFF; color: #0A0A0A; border-color: #FFFFFF; }
.bar { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px 10px; margin-top: 6px; }
.hint { flex: 1 1 240px; min-height: 5.4em; font-size: 13.5px; line-height: 1.35; color: #9A9A9A; }
.tipd { position: absolute; pointer-events: none; opacity: 0; z-index: 3;
        background: rgba(22,22,22,.97); border: 1px solid #6E6E6E; border-radius: 9px;
        padding: 6px 10px; font-size: 13.5px; transition: opacity .12s ease; }
.tipd b { display: block; font-size: 14.5px; }
.tipd span { color: #D2D2D2; }
@media (prefers-reduced-motion: reduce) { .tiles { display: none; } }
"""

JS = r"""
const NS = "http://www.w3.org/2000/svg";
const C = 500, R = 480;
const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Complex numbers.
const cx = (re, im) => ({ re, im });
const add = (a, b) => cx(a.re + b.re, a.im + b.im);
const mul = (a, b) => cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const scale = (a, s) => cx(a.re * s, a.im * s);
const conj = (a) => cx(a.re, -a.im);
const abs = (a) => Math.hypot(a.re, a.im);
const div = (a, b) => { const d = b.re * b.re + b.im * b.im;
  return cx((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d); };
const ZERO = cx(0, 0), ONE = cx(1, 0);

// Isometries of the hyperbolic plane in the Poincare disk: Mobius maps
// z -> (a z + b) / (conj(b) z + conj(a)) with |a|^2 - |b|^2 = 1.
function normal(m) {
  const d = Math.sqrt(Math.max(abs(m.a) ** 2 - abs(m.b) ** 2, 1e-300));
  return { a: scale(m.a, 1 / d), b: scale(m.b, 1 / d) };
}
const ID = { a: ONE, b: ZERO };
const apply = (m, z) => div(add(mul(m.a, z), m.b), add(mul(conj(m.b), z), conj(m.a)));
// m2 after m1.
const then = (m2, m1) => normal({ a: add(mul(m2.a, m1.a), mul(m2.b, conj(m1.b))),
                                  b: add(mul(m2.a, m1.b), mul(m2.b, conj(m1.a))) });
const toOrigin = (p) => normal({ a: ONE, b: scale(p, -1) });   // translation taking p to 0
const fromOrigin = (p) => normal({ a: ONE, b: p });            // translation taking 0 to p
const halfTurn = (m) => then(fromOrigin(m), then({ a: cx(0, 1), b: ZERO }, toOrigin(m)));
const scr = (z) => [C + R * z.re, C - R * z.im];
const f2 = (v) => v.toFixed(2);

// The geodesic between two points is an arc of the circle through them that
// meets the rim at right angles (or a straight line through the centre).
// That circle's centre and radius, in disk units, or null if it's straight.
function geoCircle(p, q) {
  if (Math.abs(p.re * q.im - p.im * q.re) < 1e-9) return null;
  const o = abs(p) >= abs(q) ? p : q;
  const r2 = o.re * o.re + o.im * o.im;
  if (r2 < 1e-12) return null;
  const ix = o.re / r2, iy = o.im / r2;
  const ax = p.re, ay = p.im, bx = q.re, by = q.im;
  const d = 2 * (ax * (by - iy) + bx * (iy - ay) + ix * (ay - by));
  if (Math.abs(d) < 1e-12) return null;
  const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = ix * ix + iy * iy;
  const ux = (a2 * (by - iy) + b2 * (iy - ay) + c2 * (ay - by)) / d;
  const uy = (a2 * (ix - bx) + b2 * (ax - ix) + c2 * (bx - ax)) / d;
  const rad = Math.hypot(ax - ux, ay - uy);
  if (!isFinite(rad) || rad * R > 1e6) return null;
  return { u: cx(ux, uy), rad };
}

function geodesic(p, q) {
  const P = scr(p), Q = scr(q);
  const circle = geoCircle(p, q);
  if (!circle) return `M${f2(P[0])} ${f2(P[1])}L${f2(Q[0])} ${f2(Q[1])}`;
  const U = scr(circle.u), rad = circle.rad * R;
  const cross = (P[0] - U[0]) * (Q[1] - U[1]) - (P[1] - U[1]) * (Q[0] - U[0]);
  return `M${f2(P[0])} ${f2(P[1])}A${f2(rad)} ${f2(rad)} 0 0 ${cross > 0 ? 1 : 0} ${f2(Q[0])} ${f2(Q[1])}`;
}

// The geodesic from p to q as a cubic Bezier's two inner control points
// (screen units): the standard close fit to a circular arc, with the
// tangents at both ends kept. The morph slides them onto a straight line.
function geoControls(p, q) {
  const P = scr(p), Q = scr(q);
  const circle = geoCircle(p, q);
  if (!circle) return [[P[0] + (Q[0] - P[0]) / 3, P[1] + (Q[1] - P[1]) / 3],
                       [P[0] + (2 * (Q[0] - P[0])) / 3, P[1] + (2 * (Q[1] - P[1])) / 3]];
  const U = scr(circle.u);
  let turn = Math.atan2(Q[1] - U[1], Q[0] - U[0]) - Math.atan2(P[1] - U[1], P[0] - U[0]);
  if (turn > Math.PI) turn -= 2 * Math.PI;
  if (turn < -Math.PI) turn += 2 * Math.PI;
  const k = (4 / 3) * Math.tan(turn / 4);
  return [[P[0] - k * (P[1] - U[1]), P[1] + k * (P[0] - U[0])],
          [Q[0] + k * (Q[1] - U[1]), Q[1] - k * (Q[0] - U[0])]];
}

// A regular {p,q} tiling of the hyperbolic plane around the origin: p-gons,
// q at each corner. Each tile is the image of the central one under a chain
// of half-turns about edge midpoints, which are symmetries of the tiling.
function makeTiling(p, q, depth) {
  const rh = Math.acosh(1 / (Math.tan(Math.PI / p) * Math.tan(Math.PI / q)));
  const rho = Math.acosh(Math.cos(Math.PI / q) / Math.sin(Math.PI / p));
  const rv = Math.tanh(rh / 2), rm = Math.tanh(rho / 2);
  const verts = [], mids = [];
  for (let k = 0; k < p; k++) {
    const a = (2 * Math.PI * k) / p + Math.PI / 2;
    verts.push(cx(rv * Math.cos(a), rv * Math.sin(a)));
    const b = a + Math.PI / p;
    mids.push(cx(rm * Math.cos(b), rm * Math.sin(b)));
  }
  const turns = mids.map(halfTurn);
  const key = (z) => `${Math.round(z.re * 1e6)},${Math.round(z.im * 1e6)}`;  // String(-0) is "0"
  const seen = new Set([key(ZERO)]);
  const tiles = [ID];
  let frontier = [ID];
  for (let level = 0; level < depth; level++) {
    const next = [];
    for (const P of frontier) {
      for (const G of turns) {
        const Q = then(P, G);
        const c = apply(Q, ZERO);
        const k = key(c);
        if (seen.has(k) || abs(c) > 0.9993) continue;
        seen.add(k); tiles.push(Q); next.push(Q);
      }
    }
    frontier = next;
  }
  const edges = [], done = new Set();
  for (const P of tiles) {
    for (let k = 0; k < p; k++) {
      const k2 = key(apply(P, mids[k]));
      if (done.has(k2)) continue;
      done.add(k2);
      edges.push([apply(P, verts[k]), apply(P, verts[(k + 1) % p])]);
    }
  }
  return { turns, centres: turns.map((G) => apply(G, ZERO)), edges };
}

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
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
  const tilePath = el("path", { class: "tiles" }, svg);
  const gRings = el("g", {}, svg);
  D.rings.forEach((ring) => {
    el("circle", { cx: C, cy: C, r: ring.radius * R, class: "ring" }, gRings);
    const t = el("text", { x: C + ring.radius * R + 6, y: C - 6, class: "ring-label" }, gRings);
    t.textContent = `${ring.ohm} Ω`;
  });
  const gEdges = el("g", {}, svg), gDots = el("g", {}, svg), gNodes = el("g", {}, svg);

  const hud = document.createElement("div");
  hud.className = "hud";
  hud.innerHTML = '<div class="t"></div><div class="d"></div>';
  wrap.appendChild(hud);
  const modes = document.createElement("div");
  modes.className = "modes";
  modes.innerHTML = '<button type="button" data-mode="hyp" class="on">Hyperbolic</button>' +
                    '<button type="button" data-mode="flat">Flat</button>';
  wrap.appendChild(modes);
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = '<button type="button" class="go-bus"></button>' +
                  '<button type="button" class="go-grid">Go to the 110 kV grid</button>' +
                  '<div class="hint"></div>';
  wrap.appendChild(bar);
  const hint = bar.querySelector(".hint");
  const tip = document.createElement("div");
  tip.className = "tipd";
  wrap.appendChild(tip);

  const state = { id: D.id, wrap, svg, hud, modes, bar, hint, tip, tilePath, gRings,
                  nodes: [], byId: {}, edges: [], V: ID, W: ID, mix: 0, mode: "hyp",
                  focusId: null, anims: {}, drag: null, fit: data.fit || {},
                  tiling: REDUCED ? null : makeTiling(7, 3, 5) };
  const showTip = (evt, title, detail) => {
    if (state.drag && state.drag.moved) return;
    const box = wrap.getBoundingClientRect();
    tip.replaceChildren();
    const b = document.createElement("b"); b.textContent = title; tip.appendChild(b);
    const s = document.createElement("span"); s.textContent = detail || ""; tip.appendChild(s);
    tip.style.opacity = 1;
    let x = evt.clientX - box.left + 14;
    if (x + tip.offsetWidth > box.width - 4) x = evt.clientX - box.left - tip.offsetWidth - 12;
    tip.style.left = `${x}px`; tip.style.top = `${evt.clientY - box.top + 14}px`;
  };
  const hideTip = () => { tip.style.opacity = 0; };
  state.hideTip = hideTip;

  D.nodes.forEach((n) => {
    const g = el("g", { class: `node ${n.kind}`, "data-id": n.id }, gNodes);
    const hit = el("circle", { class: "hit", r: 10 }, g);  // easier to click than the tile
    const shape = n.kind === "root" ? el("circle", { r: 14 }, g) : el("rect", { rx: 4 }, g);
    const label = el("text", {}, g);
    const node = { n, g, hit, shape, label, base: cx(n.x, n.y), flat0: cx(n.fx, n.fy),
                   flat: cx(n.fx, n.fy), disp: cx(n.x, n.y), screen: [C, C] };
    state.nodes.push(node);
    state.byId[String(n.id)] = node;
    g.addEventListener("mousemove", (evt) => showTip(evt, n.label, n.detail));
    g.addEventListener("mouseleave", hideTip);
  });
  D.edges.forEach((e) => {
    const path = el("path", { class: "geo" }, gEdges);
    const dots = el("path", { class: "dots" }, gDots);
    state.edges.push({ a: state.byId[String(e.a)], b: state.byId[String(e.b)], path, dots,
                       phase: Math.random() * 13, speed: 0, target: 0 });
  });

  bar.querySelector(".go-grid").addEventListener("click", () => travel(state, "grid"));
  bar.querySelector(".go-bus").addEventListener("click", () => travel(state, state.selected));
  modes.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => setMode(state, b.dataset.mode)));

  // Drag to glide: the point under the pointer follows it, by the
  // hyperbolic translation taking where it was to where it is now.
  const toDisk = (evt) => {
    const rect = svg.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) * 1000) / rect.width;
    const y = ((evt.clientY - rect.top) * 1000) / rect.height;
    return cx((x - C) / R, (C - y) / R);
  };
  svg.addEventListener("pointerdown", (evt) => {
    const z = toDisk(evt);
    if (abs(z) >= 0.985) return;
    const target = evt.target.closest ? evt.target.closest(".node") : null;
    state.drag = { id: evt.pointerId, p0: z, V0: state.V, W0: state.W, moved: false,
                   x: evt.clientX, y: evt.clientY, node: target };
    try { svg.setPointerCapture(evt.pointerId); } catch (err) { /* not an active pointer */ }
  });
  svg.addEventListener("pointermove", (evt) => {
    const d = state.drag;
    if (!d || d.id !== evt.pointerId || state.mode !== "hyp") return;
    if (!d.moved && Math.hypot(evt.clientX - d.x, evt.clientY - d.y) < 5) return;
    if (!d.moved) { d.moved = true; hideTip(); svg.classList.add("dragging"); }
    if (state.anims.move) { cancelAnimationFrame(state.anims.move); state.anims.move = null; }
    let p1 = toDisk(evt);
    const r = abs(p1);
    if (r > 0.97) p1 = scale(p1, 0.97 / r);
    const move = then(fromOrigin(p1), toOrigin(d.p0));
    state.V = then(move, d.V0);
    state.W = then(move, d.W0);
    rebaseTiles(state);
    layoutAll(state);
  });
  const end = (evt) => {
    const d = state.drag;
    state.drag = null;
    svg.classList.remove("dragging");
    if (!d || d.id !== evt.pointerId) return;
    if (!d.moved && d.node) travel(state, d.node.getAttribute("data-id"));
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  return state;
}

// Keep the tiling's central tile near the middle of the view, by composing
// with tiling symmetries: the picture doesn't change, but the tiles that
// were generated stay where they're needed.
function rebaseTiles(state) {
  if (!state.tiling) return;
  for (let i = 0; i < 40; i++) {
    let best = -1, bestR = abs(apply(state.W, ZERO));
    state.tiling.centres.forEach((c, k) => {
      const r = abs(apply(state.W, c));
      if (r < bestR - 1e-9) { bestR = r; best = k; }
    });
    if (best < 0) return;
    state.W = then(state.W, state.tiling.turns[best]);
  }
}

function drawTiles(state) {
  if (!state.tiling || state.mix > 0.999) { state.tilePath.setAttribute("d", ""); return; }
  const parts = [];
  for (const [u, v] of state.tiling.edges) {
    const a = apply(state.W, u), b = apply(state.W, v);
    if (abs(a) > 0.998 && abs(b) > 0.998) continue;
    parts.push(geodesic(a, b));
  }
  state.tilePath.setAttribute("d", parts.join(""));
  state.tilePath.style.opacity = String(1 - state.mix);
}

function layoutAll(state) {
  const mix = state.mix;
  let nearest = null, nearestR = Infinity;
  for (const node of state.nodes) {
    node.disp = apply(state.V, node.base);
    const r = abs(node.disp);
    if (r < nearestR) { nearestR = r; nearest = node; }
    const sc = Math.max(0, 1 - r * r);
    node.sc = sc;
    const H = scr(node.disp), F = scr(node.flat);
    const P = [H[0] + (F[0] - H[0]) * mix, H[1] + (F[1] - H[1]) * mix];
    node.screen = P;
    node.F = F;
    const kind = node.n.kind;
    const hs = (kind === "sub" ? 38 : kind === "root" ? 30 : 18) * sc;
    const fs = kind === "sub" ? 16 : kind === "root" ? 14 : 7;
    const s = hs + (fs - hs) * mix;
    node.hit.setAttribute("cx", P[0]); node.hit.setAttribute("cy", P[1]);
    node.hit.setAttribute("r", Math.max(s / 2 + 3, 10));
    if (kind === "root") {
      node.shape.setAttribute("cx", P[0]); node.shape.setAttribute("cy", P[1]);
      node.shape.setAttribute("r", Math.max(1, s / 2));
    } else {
      node.shape.setAttribute("x", P[0] - s / 2); node.shape.setAttribute("y", P[1] - s / 2);
      node.shape.setAttribute("width", s); node.shape.setAttribute("height", s);
      node.shape.setAttribute("rx", Math.max(1, s * 0.22));
      node.shape.setAttribute("stroke-width", Math.max(0.6, 2 * (sc + (0.5 - sc) * mix)));
    }
    node.g.style.display = s < 0.5 ? "none" : "";
  }
  if (mix < 0.5) state.focusId = nearest ? String(nearest.n.id) : null;
  for (const node of state.nodes) {
    const kind = node.n.kind;
    const isFocus = String(node.n.id) === state.focusId && mix < 0.5;
    node.g.classList.toggle("focus", isFocus);
    const show = mix < 0.5 ? (kind !== "bus" ? node.sc > 0.35 : isFocus) : kind === "sub";
    node.label.textContent = show ? node.n.label : "";
    node.label.setAttribute("x", node.screen[0]);
    const s = kind === "sub" ? 38 * node.sc : 18 * node.sc;
    node.label.setAttribute("y", node.screen[1] + Math.max(s / 2, 8) + 16);
  }
  state.edges.forEach((item) => {
    const a = item.a, b = item.b;
    let d;
    if (mix === 0) {
      d = geodesic(a.disp, b.disp);
    } else if (mix >= 1) {
      d = `M${f2(a.screen[0])} ${f2(a.screen[1])}L${f2(b.screen[0])} ${f2(b.screen[1])}`;
    } else {  // part way: the arc straightens as its ends slide to the flat drawing
      const [h1, h2] = geoControls(a.disp, b.disp);
      const t1 = [a.F[0] + (b.F[0] - a.F[0]) / 3, a.F[1] + (b.F[1] - a.F[1]) / 3];
      const t2 = [a.F[0] + (2 * (b.F[0] - a.F[0])) / 3, a.F[1] + (2 * (b.F[1] - a.F[1])) / 3];
      const c1 = [h1[0] + (t1[0] - h1[0]) * mix, h1[1] + (t1[1] - h1[1]) * mix];
      const c2 = [h2[0] + (t2[0] - h2[0]) * mix, h2[1] + (t2[1] - h2[1]) * mix];
      d = `M${f2(a.screen[0])} ${f2(a.screen[1])}C${f2(c1[0])} ${f2(c1[1])} ${f2(c2[0])} ${f2(c2[1])} ` +
          `${f2(b.screen[0])} ${f2(b.screen[1])}`;
    }
    item.path.setAttribute("d", d);
    item.dots.setAttribute("d", d);
    const mid = cx((a.disp.re + b.disp.re) / 2, (a.disp.im + b.disp.im) / 2);
    const sc = Math.max(0, 1 - abs(mid) ** 2);
    const w = 5 * sc + (2 - 5 * sc) * mix;
    item.path.setAttribute("stroke-width", Math.max(0.5, w));
    item.dots.setAttribute("stroke-width", Math.max(0, 6 * sc * (1 - mix) + 3 * mix));
    item.dots.style.display = sc < 0.05 && mix < 0.5 ? "none" : "";
  });
  state.gRings.style.opacity = String(1 - mix);
  drawTiles(state);

  const t = state.hud.querySelector(".t"), d = state.hud.querySelector(".d");
  if (mix >= 0.5) {
    const fit = state.fit;
    t.textContent = "Flat drawing";
    d.textContent = fit.flat != null
      ? `Distances off by ${Math.round(fit.flat * 100)}% on average, against ${Math.round(fit.curved * 100)}% in hyperbolic space.`
      : "";
    state.hint.textContent = "The same grid on a flat page, fitted to the same electrical " +
      "distances. The branches run out of room, so buses that are electrically close get pushed apart.";
  } else {
    const f = state.byId[state.focusId];
    t.textContent = f ? `Centre: ${f.n.label}` : "";
    d.textContent = f && f.n.detail ? f.n.detail : "";
    state.hint.textContent = "Drag to glide through the grid, or click a bus to travel to it.";
  }
}

// Two independent animations: "move" (the view travels) and "morph"
// (hyperbolic <-> flat), so one never leaves the other half done.
function animate(state, slot, ms, frame, done) {
  if (state.anims[slot]) cancelAnimationFrame(state.anims[slot]);
  const t0 = performance.now();
  const step = (now) => {
    // A frame's timestamp can be a little before t0: clamp, or a 1 ms
    // (reduced motion) animation would overshoot wildly.
    const t = Math.min(1, Math.max(0, (now - t0) / ms));
    const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    frame(e);
    if (t < 1) state.anims[slot] = requestAnimationFrame(step);
    else { state.anims[slot] = null; if (done) done(); }
  };
  state.anims[slot] = requestAnimationFrame(step);
}

// Move the view so a node comes to the centre, along the hyperbolic
// straight line, at an even hyperbolic speed.
function travel(state, id) {
  state.hideTip();
  const node = state.byId[String(id)];
  if (!node) return;
  if (state.mode !== "hyp") setMode(state, "hyp");
  const w = apply(state.V, node.base);
  const len = abs(w);
  if (len < 1e-9) return;
  const reach = Math.atanh(Math.min(len, 1 - 1e-12));  // half the hyperbolic distance
  const V0 = state.V, W0 = state.W;
  const ms = REDUCED ? 1 : Math.min(2600, 600 + 520 * reach);  // longer trips take longer
  animate(state, "move", ms, (e) => {
    const u = scale(w, Math.tanh(e * reach) / len);
    const move = toOrigin(u);
    state.V = then(move, V0);
    state.W = then(move, W0);
    rebaseTiles(state);
    layoutAll(state);
  });
}

// Turn the flat drawing (mirrored too, if that fits better) to line up with
// the hyperbolic view on screen, so the morph moves every bus as little as
// it can. Turning and mirroring keep every flat distance.
function alignFlat(state) {
  // Each bus counts by its size on screen, so what's in view matters most.
  let total = 0, mx = 0, my = 0, fmx = 0, fmy = 0;
  for (const node of state.nodes) {
    const w = Math.max(0, 1 - abs(node.disp) ** 2) + 1e-3;
    node.weight = w;
    total += w;
    mx += w * node.disp.re; my += w * node.disp.im;
    fmx += w * node.flat0.re; fmy += w * node.flat0.im;
  }
  mx /= total; my /= total; fmx /= total; fmy /= total;
  let best = null;
  for (const mirror of [1, -1]) {
    let c = 0, s = 0;
    for (const node of state.nodes) {
      const fx = node.flat0.re - fmx, fy = (node.flat0.im - fmy) * mirror;
      const hx = node.disp.re - mx, hy = node.disp.im - my;
      c += node.weight * (fx * hx + fy * hy);
      s += node.weight * (fx * hy - fy * hx);
    }
    const score = Math.hypot(c, s);
    if (!best || score > best.score) best = { score, mirror, angle: Math.atan2(s, c) };
  }
  const co = Math.cos(best.angle), si = Math.sin(best.angle);
  for (const node of state.nodes) {
    const fx = node.flat0.re, fy = node.flat0.im * best.mirror;
    node.flat = cx(fx * co - fy * si, fx * si + fy * co);
  }
}

function setMode(state, mode, done) {
  state.modes.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  if (state.mode === mode) { if (done) done(); return; }
  if (mode === "flat" && state.mix === 0) alignFlat(state);
  state.mode = mode;
  const from = state.mix, to = mode === "flat" ? 1 : 0;
  animate(state, "morph", REDUCED ? 1 : 1400,
          (e) => { state.mix = from + (to - from) * e; layoutAll(state); }, done);
}

function update(state, data) {
  state.fit = data.fit || state.fit;
  state.selected = String(data.selected);
  const sel = state.byId[state.selected];
  const goBus = state.bar.querySelector(".go-bus");
  goBus.textContent = sel ? (sel.n.kind === "bus" ? `Go to bus ${sel.n.id}` : `Go to ${sel.n.label}`) : "";
  goBus.style.display = sel ? "" : "none";
  for (const node of state.nodes) {
    const st = data.states[String(node.n.id)];
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
  layoutAll(state);
}

function tick(state, now) {
  if (!state.wrap.isConnected) { state.running = false; return; }
  const dt = Math.min(Math.max(now - state.last, 0) / 1000, 0.1);
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
    const start = state.byId[String(data.start)];
    if (start) {
      state.V = toOrigin(start.base);
      state.W = state.V;
      rebaseTiles(state);
    }
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
