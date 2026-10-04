"""A 3D globe for the From orbit view: drag it to turn it, and watch a
polar-orbiting satellite, like Terra or Aqua, circle the Earth. Each time it
passes over the grid in Germany, the grid's dot ripples like a pond.

The orbit and its speed are for illustration: the orbit is tilted 98.2
degrees like Terra's and Aqua's, but a real one passes over a given place
about twice a day, not on every lap. Land outlines: Natural Earth (public
domain), stored in results/world_land.json so the app runs offline. Plain
JavaScript drawing on a canvas, no libraries.
"""
import json
from pathlib import Path

import streamlit as st

WORLD_FILE = Path(__file__).parent / "results" / "world_land.json"
INCLINATION = 98.2   # degrees, Terra's and Aqua's sun-synchronous orbits

CSS = """
.gl { position: relative; width: 100%; font-family: "Segoe UI Variable Text", "Segoe UI",
      system-ui, sans-serif; }
.gl canvas { display: block; margin: 0 auto; cursor: grab; touch-action: none; outline: none;
             border-radius: 50%; }
.gl canvas:active { cursor: grabbing; }
.gl canvas:focus-visible { box-shadow: 0 0 0 2px #FFD34D; }
.gl .bar { display: flex; align-items: center; justify-content: center; gap: 14px;
           margin: 6px 0 2px; min-height: 30px; }
.gl .status { font-size: 13px; color: #ADADAD; transition: color .3s; }
.gl .status.on { color: #FFD34D; }
.gl button { font: inherit; font-size: 13px; color: #FFFFFF; background: transparent;
             border: 1.2px solid #8A8A8A; border-radius: 10px; padding: 4px 12px; cursor: pointer; }
.gl button:hover { border-color: #FFFFFF; background: #1A1A1A; }
.gl button:focus-visible { outline: 2px solid #FFD34D; outline-offset: 2px; }
"""

JS = r"""
const DEG = Math.PI / 180;
const REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const ALT = 1.11;            // orbit radius in Earth radii: about 705 km up, like Terra and Aqua
const LAP = REDUCED ? 30 : 9;   // seconds per lap, sped up for illustration
const SPIN = REDUCED ? 0 : 5;   // degrees per second the globe turns by itself
const RIPPLE = 2.1;          // seconds each ripple ring lives
const SUN = "#FFD34D";

const vec = (lat, lon) => [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
const comb = (a, s, b, t) => [a[0] * s + b[0] * t, a[1] * s + b[1] * t, a[2] * s + b[2] * t];
const clampLat = (v) => Math.max(-70, Math.min(80, v));
const ease = (x) => 1 - Math.pow(1 - x, 3);

function build(host, data) {
  host.querySelectorAll(".gl").forEach((n) => n.remove());
  const wrap = document.createElement("div");
  wrap.className = "gl";
  const canvas = document.createElement("canvas");
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "A globe you can drag to turn. A polar-orbiting satellite " +
    "circles it, and each time it passes over the grid in Germany, the grid's dot ripples.");
  wrap.appendChild(canvas);
  const bar = document.createElement("div");
  bar.className = "bar";
  bar.innerHTML = '<span class="status">Drag to turn the globe</span>' +
                  '<button type="button">Face the grid</button>';
  wrap.appendChild(bar);
  host.appendChild(wrap);

  const [glat, glon] = data.grid;
  const la = glat * DEG, lo = glon * DEG;
  const P = vec(la, lo);
  const E = [-Math.sin(lo), Math.cos(lo), 0];
  const N = [-Math.sin(la) * Math.cos(lo), -Math.sin(la) * Math.sin(lo), Math.cos(la)];
  // The orbit's direction over the grid, for a great circle tilted like Terra's.
  const head = Math.asin(Math.cos(data.incl * DEG) / Math.cos(la));
  const A = comb(N, Math.cos(head), E, Math.sin(head));
  const graticule = [];
  for (let lat = -60; lat <= 60; lat += 30) {
    const line = []; for (let lon = -180; lon <= 180; lon += 3) line.push(vec(lat * DEG, lon * DEG));
    graticule.push(line);
  }
  for (let lon = -180; lon < 180; lon += 30) {
    const line = []; for (let lat = -90; lat <= 90; lat += 3) line.push(vec(lat * DEG, lon * DEG));
    graticule.push(line);
  }
  const state = {
    id: data.id, wrap, canvas, status: bar.querySelector(".status"),
    rings: data.rings.map((r) => r.map(([x, y]) => vec(y * DEG, x * DEG))),
    graticule, P, E, N, A, glat, glon,
    lat0: 30, lon0: glon + 35, theta: -2.4, drag: null, holdUntil: 0, spinIn: 1,
    aim: null, ripples: [], statusUntil: 0, size: 0, dpr: 1, last: performance.now(), running: false,
  };

  const resize = () => {
    const size = Math.max(200, Math.min(wrap.clientWidth || 300, 360));
    const dpr = window.devicePixelRatio || 1;
    if (size === state.size && dpr === state.dpr) return;
    state.size = size; state.dpr = dpr;
    canvas.style.width = canvas.style.height = `${size}px`;
    canvas.width = canvas.height = Math.round(size * dpr);
  };
  resize();
  if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);

  const pause = (ms) => { state.holdUntil = performance.now() + ms; state.spinIn = 0; };
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    state.drag = { x: e.clientX, y: e.clientY, lat0: state.lat0, lon0: state.lon0 };
    state.aim = null;
  });
  canvas.addEventListener("pointermove", (e) => {
    const d = state.drag; if (!d) return;
    const k = 180 / (state.size * 0.9);   // degrees per pixel: a drag across the globe is half a turn
    state.lon0 = d.lon0 - (e.clientX - d.x) * k;
    state.lat0 = clampLat(d.lat0 + (e.clientY - d.y) * k);
  });
  const release = () => { if (state.drag) { state.drag = null; pause(2500); } };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: [0, 12], ArrowRight: [0, -12], ArrowUp: [-10, 0], ArrowDown: [10, 0] }[e.key];
    if (!step) return;
    e.preventDefault();
    state.lat0 = clampLat(state.lat0 + step[0]); state.lon0 += step[1]; state.aim = null; pause(4000);
  });
  bar.querySelector("button").addEventListener("click", () => {
    let dLon = ((state.glon - state.lon0) % 360 + 540) % 360 - 180;   // the short way round
    state.aim = { t0: performance.now(), lat: state.lat0, lon: state.lon0,
                  dLat: (state.glat - 12) - state.lat0, dLon };
    pause(7000);
  });
  return state;
}

// Whether the grid faces us (the pond is only drawn whole when it does).
const gzFront = (state) => view(state, state.P)[2] > 0.2;

// Earth-fixed unit vector (times r) to the screen: x right, y up, z toward us.
function view(state, v, r = 1) {
  const a = state.lat0 * DEG, b = state.lon0 * DEG;
  const x1 = v[0] * Math.cos(b) + v[1] * Math.sin(b);
  const y1 = -v[0] * Math.sin(b) + v[1] * Math.cos(b);
  const z1 = v[2];
  return [r * y1, r * (Math.cos(a) * z1 - Math.sin(a) * x1), r * (Math.sin(a) * z1 + Math.cos(a) * x1)];
}

function strokeFront(ctx, state, pts, C, R, r = 1) {
  ctx.beginPath();
  let pen = false;
  for (const v of pts) {
    const [x, y, z] = view(state, v, r);
    if (z > 0) {
      if (pen) ctx.lineTo(C + R * x, C - R * y); else ctx.moveTo(C + R * x, C - R * y);
      pen = true;
    } else pen = false;
  }
  ctx.stroke();
}

function draw(state, now) {
  const { canvas, size, dpr } = state;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, size, size);
  const C = size / 2, R = size * 0.4;

  // Atmosphere, ocean and the night-side shading that makes it read as a ball.
  const air = ctx.createRadialGradient(C, C, R * 0.98, C, C, R * 1.1);
  air.addColorStop(0, "rgba(110,170,240,0.35)"); air.addColorStop(1, "rgba(110,170,240,0)");
  ctx.fillStyle = air; ctx.beginPath(); ctx.arc(C, C, R * 1.1, 0, 2 * Math.PI); ctx.fill();
  const sea = ctx.createRadialGradient(C - R * 0.3, C - R * 0.35, R * 0.1, C, C, R);
  sea.addColorStop(0, "#1B3D5C"); sea.addColorStop(1, "#08131E");
  ctx.fillStyle = sea; ctx.beginPath(); ctx.arc(C, C, R, 0, 2 * Math.PI); ctx.fill();

  ctx.save();
  ctx.beginPath(); ctx.arc(C, C, R, 0, 2 * Math.PI); ctx.clip();
  ctx.strokeStyle = "rgba(90,140,190,0.22)"; ctx.lineWidth = 0.7;
  for (const line of state.graticule) strokeFront(ctx, state, line, C, R);
  // Land: points on the far side slide to the edge, so shapes that cross it stay closed.
  ctx.beginPath();
  for (const ring of state.rings) {
    let seen = 0, first = true;
    const pts = ring.map((v) => {
      let [x, y, z] = view(state, v);
      if (z > 0) seen++; else { const n = Math.hypot(x, y) || 1; x /= n; y /= n; }
      return [C + R * x, C - R * y];
    });
    if (seen < 2) continue;
    for (const [x, y] of pts) { if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y); }
    ctx.closePath();
  }
  ctx.fillStyle = "#33452F"; ctx.fill();
  ctx.strokeStyle = "#6E8A68"; ctx.lineWidth = 0.6; ctx.stroke();
  const shade = ctx.createRadialGradient(C - R * 0.4, C - R * 0.45, R * 0.05, C - R * 0.1, C - R * 0.1, R * 1.25);
  shade.addColorStop(0, "rgba(255,255,255,0.10)"); shade.addColorStop(0.5, "rgba(0,0,0,0)");
  shade.addColorStop(1, "rgba(0,0,0,0.5)");
  ctx.fillStyle = shade; ctx.fillRect(C - R, C - R, 2 * R, 2 * R);

  // The ground track: where the satellite passes over.
  const track = [];
  for (let k = 0; k <= 180; k++) { const t = k * 2 * DEG; track.push(comb(state.P, Math.cos(t), state.A, Math.sin(t))); }
  ctx.setLineDash([3, 4]); ctx.strokeStyle = "rgba(156,200,240,0.35)"; ctx.lineWidth = 1;
  strokeFront(ctx, state, track, C, R);
  ctx.setLineDash([]);

  // Ripples on the surface around the grid, like a stone dropped in a yellow pond.
  const [gx0, gy0, gz] = view(state, state.P);
  const gx = C + R * gx0, gy = C - R * gy0;
  let glow = 0;
  for (const rp of state.ripples) {
    const age = (now - rp) / 1000;
    glow = Math.max(glow, 1 - age / 1.1);
    for (let k = 0; k < 3; k++) {
      const t = age - 0.32 * k;
      if (t <= 0 || t >= RIPPLE) continue;
      const f = t / RIPPLE;
      const ang = (0.6 + 16 * ease(f)) * DEG;
      const ring = [];
      for (let j = 0; j <= 72; j++) {
        const b = j * 5 * DEG;
        const dir = comb(state.E, Math.cos(b), state.N, Math.sin(b));
        ring.push(comb(state.P, Math.cos(ang), dir, Math.sin(ang)));
      }
      if (k === 0 && gzFront(state)) {   // the pond: a soft yellow wash inside the first ring
        ctx.beginPath();
        ring.forEach((v, j) => { const [x, y] = view(state, v); j ? ctx.lineTo(C + R * x, C - R * y) : ctx.moveTo(C + R * x, C - R * y); });
        ctx.fillStyle = `rgba(255,211,77,${(0.16 * Math.pow(1 - f, 1.3)).toFixed(3)})`;
        ctx.fill();
      }
      ctx.strokeStyle = `rgba(255,211,77,${(0.95 * Math.pow(1 - f, 1.4)).toFixed(3)})`;
      ctx.lineWidth = 2.8 - k * 0.6;
      strokeFront(ctx, state, ring, C, R);
    }
  }
  if (gz > 0) {
    if (glow > 0) {
      const pond = ctx.createRadialGradient(gx, gy, 0, gx, gy, 10 + 26 * (1 - glow));
      pond.addColorStop(0, `rgba(255,211,77,${(0.55 * glow).toFixed(3)})`);
      pond.addColorStop(1, "rgba(255,211,77,0)");
      ctx.fillStyle = pond; ctx.beginPath(); ctx.arc(gx, gy, 10 + 26 * (1 - glow), 0, 2 * Math.PI); ctx.fill();
    }
    ctx.shadowColor = SUN; ctx.shadowBlur = 6 + 14 * Math.max(glow, 0);
    ctx.fillStyle = SUN; ctx.beginPath(); ctx.arc(gx, gy, 3.4 + 2.6 * Math.max(glow, 0), 0, 2 * Math.PI); ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.restore();

  // The orbit: bright in front of the Earth, faint behind it, hidden where the Earth covers it.
  const orbit = [];
  for (let k = 0; k <= 180; k++) { const t = k * 2 * DEG; orbit.push(comb(state.P, Math.cos(t), state.A, Math.sin(t))); }
  for (const front of [false, true]) {
    ctx.beginPath(); let pen = false;
    for (const v of orbit) {
      const [x, y, z] = view(state, v, ALT);
      const show = front ? z > 0 : (z <= 0 && Math.hypot(x, y) > 1);
      if (show) { if (pen) ctx.lineTo(C + R * x, C - R * y); else ctx.moveTo(C + R * x, C - R * y); pen = true; }
      else pen = false;
    }
    ctx.strokeStyle = front ? "rgba(156,200,240,0.75)" : "rgba(156,200,240,0.2)";
    ctx.lineWidth = front ? 1.2 : 1; ctx.stroke();
  }

  // The satellite, and its line of sight while it passes over the grid.
  const th = state.theta;
  const sv = comb(state.P, Math.cos(th), state.A, Math.sin(th));
  const [sx0, sy0, sz] = view(state, sv, ALT);
  const visible = sz > 0 || Math.hypot(sx0, sy0) > 1;
  const sx = C + R * sx0, sy = C - R * sy0;
  const near = Math.abs(Math.atan2(Math.sin(th), Math.cos(th)));
  if (visible && gz > 0 && near < 0.4) {
    const a = 1 - near / 0.4;
    const beam = ctx.createLinearGradient(sx, sy, gx, gy);
    beam.addColorStop(0, `rgba(220,235,255,${(0.7 * a).toFixed(3)})`);
    beam.addColorStop(1, `rgba(255,211,77,${(0.9 * a).toFixed(3)})`);
    ctx.strokeStyle = beam; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(gx, gy); ctx.stroke();
  }
  if (visible) {
    const dv = comb(state.P, -Math.sin(th), state.A, Math.cos(th));
    const [dx, dy] = view(state, dv, 1);
    ctx.save();
    ctx.translate(sx, sy); ctx.rotate(Math.atan2(-dy, dx));
    ctx.fillStyle = "#6FA8DC"; ctx.fillRect(-2, -12, 4, 8); ctx.fillRect(-2, 4, 4, 8);
    ctx.fillStyle = "#E8EEF4"; ctx.fillRect(-4, -3, 8, 6);
    ctx.restore();
  }
}

function tick(state, now) {
  if (!state.canvas.isConnected) { state.running = false; return; }
  const dt = Math.min(0.05, Math.max(0, (now - state.last) / 1000));
  state.last = now;
  if (state.aim) {
    const f = Math.min(1, (now - state.aim.t0) / 1200), e = ease(f);
    state.lat0 = state.aim.lat + state.aim.dLat * e;
    state.lon0 = state.aim.lon + state.aim.dLon * e;
    if (f >= 1) state.aim = null;
  } else if (!state.drag && now > state.holdUntil) {
    state.spinIn = Math.min(1, state.spinIn + dt);
    state.lon0 -= SPIN * state.spinIn * dt;
  }
  const before = Math.atan2(Math.sin(state.theta), Math.cos(state.theta));
  state.theta += (2 * Math.PI / LAP) * dt;
  const after = Math.atan2(Math.sin(state.theta), Math.cos(state.theta));
  if (before < 0 && after >= 0) {          // directly over the grid
    state.ripples.push(now);
    state.statusUntil = now + 2600;
  }
  state.ripples = state.ripples.filter((t) => now - t < (RIPPLE + 1) * 1000);
  const on = now < state.statusUntil;
  if (on !== state.statusOn) {
    state.statusOn = on;
    state.status.textContent = on ? "Over the grid now: measuring its sunlight" : "Drag to turn the globe";
    state.status.classList.toggle("on", on);
  }
  draw(state, now);
  requestAnimationFrame((t) => tick(state, t));
}

export default function (component) {
  const { data, parentElement } = component;
  if (!data || !data.rings) return;
  let state = parentElement.__headroomGlobe;
  if (!state || state.id !== data.id || !state.canvas.isConnected) {
    state = build(parentElement, data);
    parentElement.__headroomGlobe = state;
  }
  if (!state.running) {
    state.running = true;
    state.last = performance.now();
    requestAnimationFrame((t) => tick(state, t));
  }
}
"""

# The version in the name makes browsers load this code fresh instead of a cached older copy.
_globe = st.components.v2.component("headroom_globe_v1", css=CSS, js=JS)


@st.cache_data
def world_rings(changed):
    """Natural Earth's land outlines as lists of [longitude, latitude]."""
    return json.loads(WORLD_FILE.read_text())["rings"]


def show_globe(grid_lat, grid_lon, key="globe"):
    """Draw the globe, with the grid at (grid_lat, grid_lon)."""
    _globe(key=key, data={"id": "globe-1", "rings": world_rings(WORLD_FILE.stat().st_mtime),
                          "grid": [grid_lat, grid_lon], "incl": INCLINATION})
