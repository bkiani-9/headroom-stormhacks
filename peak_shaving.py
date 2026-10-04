"""The 3% rule: how big a solar project each bus can take when the grid
operator may curtail up to 3% of its yearly energy.

Germany's Energy Industry Act lets grid operators plan on the assumption
that the forecast yearly energy of each wind or solar plant may be cut by
up to 3% ("Spitzenkappung", section 11(2) EnWG). This script asks, for every
bus: what is the largest project whose curtailment over a real year stays
within that 3%?

The year is 2024 at the grid's location, hour by hour, from NASA satellite
sunlight (results/sunlight_2024.csv, from fetch_sunlight.py). The same
conventions as sunlight.py's year check:
  - sun share s = the hour's sunlight / the year's brightest hour;
  - the project produces size x s, and the solar already on the grid 0.8 x s;
  - load stays at 10% of peak, its minimum (the worst case for voltage),
    so the answer errs on the safe side.

For every bus and every daylight hour, the ML model's capacity search
(ml.model_capacity_bisect) finds the most the bus can take in that hour.
Anything the project would produce above that is curtailed. The 3% size is
the largest size, to 0.1 MW, whose curtailed energy is at most 3% of what it
would produce in the year. It is never below the firm capacity
(results/capacity.csv): a firm project fits even at the brightest hour, so
it is never curtailed.

Then the real power flow rechecks every bus where the rule changes the
answer. For each, find_capacity() finds the capacity at a spread of the
hours where the model curtails, plus a few hours just below them; the
curtailment is recomputed from the solver's capacities (linear in the sun
share between those hours). If the solver puts it over 3%, the size comes
down 0.1 MW at a time until it doesn't. The saved size is the rechecked one.

Run from the project folder (after train_model.py and fetch_sunlight.py):

    py -3.14 peak_shaving.py

Saves results/peak_shaving.csv, one row per bus:
  bus, firm_mw, rule_mw (rechecked), model_rule_mw (before the recheck),
  curtailed_share (of the yearly energy at rule_mw), curtailed_hours,
  firm_mwh and rule_mwh (energy delivered in a year), gain_mwh
  (rule_mwh - firm_mwh), checked_by
and results/peak_shaving_check.json, a summary of the power flow recheck.
"""
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import ml
import sunlight
from capacity import MAX_SEARCH_MW, find_capacity
from capacity_data import read_capacity
from engine import load_grid

ROOT = Path(__file__).parent
OUT_FILE = ROOT / "results" / "peak_shaving.csv"
CHECK_FILE = ROOT / "results" / "peak_shaving_check.json"
SHARE = 0.03          # section 11(2) EnWG: up to 3% of the yearly energy
STEP_MW = 0.1
LOAD_LEVEL = sunlight.LOAD_LEVEL  # 10% of peak, all year
GEN_PEAK = sunlight.GEN_PEAK      # the solar already on the grid, at the brightest hour
N_CHECK_HOURS = 10    # binding hours rechecked per bus
N_BELOW = 3           # extra hours just below the binding ones
WORKERS = 2


def hourly_capacity(models, bus_features, buses, sun):
    """ML capacity (MW) of every bus in every hour: array [bus, hour]."""
    cases = pd.DataFrame({
        "bus": np.repeat(buses, len(sun)),
        "load_level": LOAD_LEVEL,
        "gen_level": np.tile(GEN_PEAK * sun, len(buses)),
    })
    caps = ml.model_capacity_bisect(models, bus_features, cases, max_mw=MAX_SEARCH_MW)
    return caps.to_numpy().reshape(len(buses), len(sun))


def curtailment(size, caps, sun):
    """(MWh curtailed in the year, MWh the project would produce uncurtailed)
    for a project of `size` MW, given each hour's capacity `caps`."""
    output = size * sun
    return float(np.maximum(output - caps, 0.0).sum()), float(output.sum())


def rule_size(caps, sun, firm):
    """The largest size (0.1 MW steps, at least `firm`) whose curtailment is at
    most SHARE of its yearly energy, with that share and the hours curtailed."""
    sizes = np.round(np.arange(0, MAX_SEARCH_MW + STEP_MW / 2, STEP_MW), 1)
    lost = np.maximum(sizes[:, None] * sun[None, :] - caps[None, :], 0.0).sum(axis=1)
    produced = sizes * sun.sum()
    share = np.divide(lost, produced, out=np.zeros_like(lost), where=produced > 0)
    ok = share <= SHARE + 1e-12
    # The share only grows with size, so the sizes that pass are 0 up to some limit.
    best = float(sizes[ok].max()) if ok.any() else 0.0
    best = max(best, float(firm))
    lost_mwh, produced_mwh = curtailment(best, caps, sun)
    hours = int((best * sun > caps + 1e-9).sum())
    return best, (lost_mwh / produced_mwh if produced_mwh else 0.0), hours, lost_mwh, produced_mwh


# ---------------------------------------------------------------- recheck

_net = None


def _start_worker():
    global _net
    warnings.filterwarnings("ignore")  # pandapower's deprecation noise
    _net = load_grid()


def _solver_capacity(task):
    """find_capacity() for one bus at one hour's conditions."""
    bus, s = task
    _net.load["scaling"] = LOAD_LEVEL
    _net.sgen["scaling"] = GEN_PEAK * s
    capacity, _ = find_capacity(_net, int(bus))
    return capacity


def check_hours(caps, sun, size):
    """Indexes of the hours to recheck at one bus: a spread of the hours the
    model curtails (always including the brightest), and N_BELOW hours just
    below the dimmest of them."""
    binding = np.flatnonzero(size * sun > caps + 1e-9)
    if binding.size == 0:
        return np.array([], dtype=int)
    order = binding[np.argsort(sun[binding])]
    picks = order[np.unique(np.linspace(0, len(order) - 1, N_CHECK_HOURS).round().astype(int))]
    dimmest = sun[order[0]]
    below = np.flatnonzero(sun < dimmest - 1e-9)
    below = below[np.argsort(-sun[below])]  # brightest first
    # Spread the extra hours over the band just below the binding ones.
    targets = dimmest - 0.04 * np.arange(1, N_BELOW + 1)
    extra = [below[np.argmin(np.abs(sun[below] - t))] for t in targets if below.size]
    return np.unique(np.concatenate([picks, np.array(extra, dtype=int)]))


def solver_share(size, sun, points_s, points_cap):
    """Curtailed share of the yearly energy for `size`, from the solver's
    capacities at a few sun shares, linear in between. Hours dimmer than
    the dimmest checked one count as uncurtailed; returns None if that isn't
    safe (the project would already exceed the solver's capacity there)."""
    order = np.argsort(points_s)
    ps, pc = np.asarray(points_s)[order], np.asarray(points_cap)[order]
    if size * ps[0] > pc[0] + 1e-9:
        return None
    inside = sun >= ps[0]
    caps = np.full(len(sun), np.inf)
    caps[inside] = np.interp(sun[inside], ps, pc)
    lost, produced = curtailment(size, caps, sun)
    return lost / produced


def recheck(table, caps_by_bus, sun):
    """Recheck every bus whose 3% size is above its firm capacity with the
    real power flow, lowering the size where the solver puts curtailment
    over 3%. Updates `table` in place; returns a summary."""
    buses = table.index[table.rule_mw > table.firm_mw + 1e-9].to_list()
    plan = {b: check_hours(caps_by_bus[b], sun, table.rule_mw[b]) for b in buses}
    tasks = [(b, float(sun[h])) for b in buses for h in plan[b]]
    print(f"Recheck: {len(tasks)} hourly capacities at {len(buses)} buses with the real power flow...")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_start_worker) as pool:
        solved = list(pool.map(_solver_capacity, tasks, chunksize=4))
    print(f"  done in {time.perf_counter() - start:.0f} s")

    rows, k = [], 0
    for b in buses:
        hours = plan[b]
        solver = np.array(solved[k:k + len(hours)], dtype=float)
        k += len(hours)
        model = caps_by_bus[b][hours]
        model_size = float(table.rule_mw[b])
        model_share = float(table.curtailed_share[b])
        size = model_size
        share = solver_share(size, sun, sun[hours], solver)
        while (share is None or share > SHARE + 1e-12) and size > table.firm_mw[b] + 1e-9:
            size = round(size - STEP_MW, 1)
            share = solver_share(size, sun, sun[hours], solver)
        size = max(size, float(table.firm_mw[b]))
        if share is None:  # can't happen at the firm size, but stay safe
            share = 0.0
        # Energy from the solver's capacities.
        order = np.argsort(sun[hours])
        inside = sun >= sun[hours][order][0]
        caps = np.full(len(sun), np.inf)
        caps[inside] = np.interp(sun[inside], sun[hours][order], solver[order])
        lost, produced = curtailment(size, caps, sun)
        table.loc[b, "rule_mw"] = size
        table.loc[b, "curtailed_share"] = round(lost / produced, 5) if produced else 0.0
        table.loc[b, "curtailed_hours"] = int((size * sun > caps + 1e-9).sum())
        table.loc[b, "rule_mwh"] = round(produced - lost, 1)
        table.loc[b, "gain_mwh"] = round(produced - lost - table.firm_mwh[b], 1)
        table.loc[b, "checked_by"] = f"solver, {len(hours)} hours"
        rows.append({
            "bus": int(b), "model_mw": model_size, "rechecked_mw": size,
            "model_share": round(model_share, 4),
            "solver_share_at_model_mw": (None if solver_share(model_size, sun, sun[hours], solver) is None
                                         else round(solver_share(model_size, sun, sun[hours], solver), 4)),
            "hours_checked": int(len(hours)),
            "capacity_gap_mw": round(float(np.mean(solver - model)), 3),
            "abs_capacity_gap_mw": round(float(np.mean(np.abs(solver - model))), 3),
        })

    lowered = [r for r in rows if r["rechecked_mw"] < r["model_mw"] - 1e-9]
    at_model = [r for r in rows if r["solver_share_at_model_mw"] is not None]
    return {
        "buses_checked": len(rows),
        "hours_checked": int(sum(r["hours_checked"] for r in rows)),
        "buses_kept": len(rows) - len(lowered),
        "buses_lowered": len(lowered),
        "most_lowered_mw": round(max((r["model_mw"] - r["rechecked_mw"] for r in lowered), default=0.0), 1),
        "mean_lowered_mw": round(float(np.mean([r["model_mw"] - r["rechecked_mw"] for r in lowered])), 2)
        if lowered else 0.0,
        "mean_capacity_gap_mw": round(float(np.mean([r["capacity_gap_mw"] for r in rows])), 3) if rows else None,
        "mean_abs_capacity_gap_mw": round(float(np.mean([r["abs_capacity_gap_mw"] for r in rows])), 3)
        if rows else None,
        "mean_solver_share_at_model_mw": round(float(np.mean([r["solver_share_at_model_mw"] for r in at_model])), 4)
        if at_model else None,
        "buses": rows,
    }


def main():
    sun_table = sunlight.read_sunlight()
    sun = sun_table.sun_share.to_numpy()
    models, bus_features = ml.load_model(), ml.read_bus_features()
    firm = read_capacity().capacity_mw
    buses = [b for b in bus_features.index if b in firm.index]
    print(f"Model: capacity of {len(buses)} buses in each of {len(sun):,} daylight hours...")
    start = time.perf_counter()
    caps = hourly_capacity(models, bus_features, buses, sun)
    print(f"  done in {time.perf_counter() - start:.0f} s")

    rows, caps_by_bus = [], {}
    for i, b in enumerate(buses):
        caps_by_bus[b] = caps[i]
        f = float(firm[b])
        size, share, hours, lost, produced = rule_size(caps[i], sun, f)
        firm_mwh = f * float(sun.sum())
        rows.append({"bus": int(b), "firm_mw": f, "rule_mw": size, "model_rule_mw": size,
                     "curtailed_share": round(share, 5), "curtailed_hours": hours,
                     "firm_mwh": round(firm_mwh, 1), "rule_mwh": round(produced - lost, 1),
                     "gain_mwh": round(produced - lost - firm_mwh, 1),
                     "checked_by": "model" if size > f + 1e-9 else "firm capacity"})
    table = pd.DataFrame(rows).set_index("bus")

    summary = recheck(table, caps_by_bus, sun)
    summary.update({
        "rule": "curtailment at most 3% of the yearly energy (section 11(2) EnWG)",
        "daylight_hours": int(len(sun)),
        "full_output_hours": round(float(sun.sum()), 1),
        "load_level": LOAD_LEVEL,
    })
    OUT_FILE.parent.mkdir(exist_ok=True)
    table.reset_index().to_csv(OUT_FILE, index=False)
    CHECK_FILE.write_text(json.dumps(summary, indent=2))
    print(f"Rechecked {summary['buses_checked']} buses: kept {summary['buses_kept']}, lowered "
          f"{summary['buses_lowered']} (at most {summary['most_lowered_mw']} MW). "
          f"Saved {OUT_FILE} and {CHECK_FILE}.")


if __name__ == "__main__":
    main()
