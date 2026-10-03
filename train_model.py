"""Train the ML model on the solved scenarios and test it.

Run it from the project folder (after make_scenarios.py and build_features.py):

    py -3.14 train_model.py

Writes:
  models/headroom_model.joblib   the trained model
  results/model_report.md        the error report
  results/model_metrics.json     the same numbers, for the app
  results/model_vs_solver.png    model-versus-solver plot
"""
import json

import matplotlib
matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

import ml
from engine import MAX_LOADING_PERCENT, MAX_VOLTAGE_PU

ROOT = ml.ROOT
SCENARIO_FILE = ROOT / "results" / "scenarios.csv"
BOUNDARY_FILE = ROOT / "results" / "scenarios_boundary.csv"
EDGE_FILE = ROOT / "results" / "scenarios_edges.csv"
CAPACITY_TEST_FILE = ROOT / "results" / "capacity_scenarios.csv"
SNAPSHOT_FILE = ROOT / "results" / "capacity.csv"
REPORT_FILE = ROOT / "results" / "model_report.md"
METRICS_FILE = ROOT / "results" / "model_metrics.json"
PLOT_FILE = ROOT / "results" / "model_vs_solver.png"
GOOD_MW = 0.5  # a capacity prediction within 0.5 MW is good (PLAN.md)

# Chart colours (one series, so one hue; references stay muted grey).
DOT = "#2a78d6"
REFERENCE = "#898781"
INK = "#52514e"


def train(x, y):
    models = {}
    for target in ml.TARGETS:
        model = HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=0.1, random_state=0,
        )
        models[target] = model.fit(x, y[target])
    return models


def pass_fail_stats(truth, predicted):
    """Agreement between true and predicted pass/fail."""
    false_pass = (predicted & ~truth).mean()  # says it fits, but it doesn't
    false_fail = (~predicted & truth).mean()  # says no, but it would fit
    return {
        "accuracy": float((truth == predicted).mean()),
        "false_pass_rate": float(false_pass),
        "false_fail_rate": float(false_fail),
    }


def main():
    # The random scenarios, plus the near-boundary and edge ones if they exist.
    files = [f for f in [SCENARIO_FILE, BOUNDARY_FILE, EDGE_FILE] if f.exists()]
    scenarios = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    scenarios = scenarios.dropna(subset=ml.TARGETS)
    print("Training data:", ", ".join(f.name for f in files))
    bus_features = ml.read_bus_features()
    x = ml.add_features(scenarios, bus_features)
    y = scenarios[ml.TARGETS]

    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
        x, y, scenarios, test_size=0.2, random_state=0)
    models = train(x_train, y_train)

    # --- Test 1: held-out scenarios ---------------------------------------
    predicted = ml.predict(models, x_test)
    errors = {}
    for target in ml.TARGETS:
        diff = (predicted[target] - y_test[target]).abs()
        errors[target] = {"mean_abs_error": float(diff.mean()),
                          "p95_abs_error": float(diff.quantile(0.95)),
                          "max_abs_error": float(diff.max())}
    truth = s_test.passes.astype(bool).to_numpy()
    model_pass = ml.passes(predicted).to_numpy()

    # Baseline: "it fits if it's no bigger than today's capacity at that bus".
    snapshot = pd.read_csv(SNAPSHOT_FILE, index_col="bus").capacity_mw
    baseline_pass = (s_test.solar_mw <= s_test.bus.map(snapshot)).to_numpy()

    # --- Test 2: capacity, against the solver's find_capacity -------------
    cap_truth = pd.read_csv(CAPACITY_TEST_FILE)
    cap_model = ml.model_capacity(models, bus_features, cap_truth)
    cap_err = cap_model - cap_truth.capacity_mw
    capacity_stats = {
        "cases": int(len(cap_truth)),
        "conditions": int(cap_truth.condition.nunique()),
        "mean_abs_error_mw": float(cap_err.abs().mean()),
        "max_abs_error_mw": float(cap_err.abs().max()),
        "share_within_0_5_mw": float((cap_err.abs() <= GOOD_MW + 1e-9).mean()),
        "share_over_by_more_than_0_5_mw": float((cap_err > GOOD_MW + 1e-9).mean()),
    }

    metrics = {
        "train_scenarios": int(len(x_train)),
        "test_scenarios": int(len(x_test)),
        "errors": errors,
        "pass_fail_model": pass_fail_stats(truth, model_pass),
        "pass_fail_baseline": pass_fail_stats(truth, baseline_pass),
        "capacity": capacity_stats,
    }

    ml.MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib_dump(models)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    write_report(metrics)
    plot(y_test, predicted, cap_truth.capacity_mw, cap_model)
    print(REPORT_FILE.read_text())


def joblib_dump(models):
    import joblib
    joblib.dump(models, ml.MODEL_FILE)


def write_report(m):
    e = m["errors"]
    pm, pb, c = m["pass_fail_model"], m["pass_fail_baseline"], m["capacity"]
    lines = [
        "# Headroom ML model: error report",
        "",
        f"Trained on {m['train_scenarios']:,} solved scenarios; "
        f"tested on {m['test_scenarios']:,} scenarios it never saw.",
        "",
        "## Predicting the solver's numbers (held-out scenarios)",
        "",
        "| Quantity | Average error | 95% of errors below | Worst error |",
        "|---|---|---|---|",
        f"| Highest voltage | {e['max_voltage_pu']['mean_abs_error']:.4f} pu "
        f"| {e['max_voltage_pu']['p95_abs_error']:.4f} pu | {e['max_voltage_pu']['max_abs_error']:.4f} pu |",
        f"| Busiest line loading | {e['max_line_loading']['mean_abs_error']:.2f}% "
        f"| {e['max_line_loading']['p95_abs_error']:.2f}% | {e['max_line_loading']['max_abs_error']:.2f}% |",
        f"| Busiest transformer loading | {e['max_trafo_loading']['mean_abs_error']:.2f}% "
        f"| {e['max_trafo_loading']['p95_abs_error']:.2f}% | {e['max_trafo_loading']['max_abs_error']:.2f}% |",
        "",
        "## Pass or fail (held-out scenarios)",
        "",
        "| | Agrees with solver | Says pass, solver fails (dangerous) | Says fail, solver passes |",
        "|---|---|---|---|",
        f"| Model | {pm['accuracy']:.1%} | {pm['false_pass_rate']:.1%} | {pm['false_fail_rate']:.1%} |",
        f"| Baseline: \"fits if no bigger than today's capacity\" | {pb['accuracy']:.1%} "
        f"| {pb['false_pass_rate']:.1%} | {pb['false_fail_rate']:.1%} |",
        "",
        f"## Capacity (against find_capacity, {c['cases']} bus-and-condition cases, "
        f"{c['conditions']} conditions)",
        "",
        f"- Average error: {c['mean_abs_error_mw']:.2f} MW; worst: {c['max_abs_error_mw']:.1f} MW",
        f"- Within 0.5 MW of the solver: {c['share_within_0_5_mw']:.1%}",
        f"- Over-predicted by more than 0.5 MW (dangerous): {c['share_over_by_more_than_0_5_mw']:.1%}",
        "",
        "The model is never trusted alone: step 6 rechecks every capacity "
        "with the real solver before it goes on the map.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(y_test, predicted, cap_true, cap_model):
    panels = [
        ("Highest voltage (pu)", y_test.max_voltage_pu, predicted.max_voltage_pu, MAX_VOLTAGE_PU),
        ("Busiest line loading (%)", y_test.max_line_loading, predicted.max_line_loading, MAX_LOADING_PERCENT),
        ("Busiest transformer loading (%)", y_test.max_trafo_loading, predicted.max_trafo_loading, MAX_LOADING_PERCENT),
        ("Capacity (MW), 177 buses x 4 corner conditions", cap_true, cap_model, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (title, truth, guess, limit) in zip(axes.flat, panels):
        low = min(truth.min(), guess.min())
        high = max(truth.max(), guess.max())
        ax.plot([low, high], [low, high], color=REFERENCE, linewidth=1,
                linestyle="--", zorder=1, label="perfect prediction")
        if limit is not None:
            ax.axvline(limit, color=REFERENCE, linewidth=1, linestyle=":", zorder=1)
            ax.axhline(limit, color=REFERENCE, linewidth=1, linestyle=":", zorder=1,
                       label="limit")
        ax.scatter(truth, guess, s=6, color=DOT, alpha=0.35, linewidths=0, zorder=2)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Solver", color=INK)
        ax.set_ylabel("Model", color=INK)
        ax.tick_params(colors=INK, labelsize=8)
        ax.grid(color="#e1e0d9", linewidth=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color("#c3c2b7")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.suptitle("Model versus solver (scenarios the model never saw)",
                 color="#0b0b0b", fontsize=13, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)


if __name__ == "__main__":
    main()
