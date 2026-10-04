# Headroom

**How much new solar can each point of a local grid take before something
breaks?**

Headroom answers that for every bus of a real-world-style distribution
grid, explains *which* limit stops each one, and shows how the answer
changes with grid conditions. It runs entirely offline on a laptop.

**Headline:** 177 of 177 buses can take a 2 MW solar project in 95% of
grid conditions (checked with the real power flow). On today's snapshot
grid, which is close to the worst case for voltage, 175 of 177 can.

## What it does

- **Project check:** pick a bus (click it on the map), drag a 0-20 MW solar
  slider, and the grid is re-solved: the map recolours by how close each
  bus, line and transformer is to its limit, and the app says pass or fail
  and names the limit that broke. A before/after switch compares the grid
  with and without the project.
- **Capacity map:** every bus coloured by its hosting capacity, the largest
  solar project it takes without breaking a limit, with the limit that
  stops it on hover. Two versions:
  - **On today's grid:** one snapshot of grid conditions.
  - **In 95% of grid conditions:** the size that fits in at least 95% of
    sampled load and generation conditions, checked with the real power
    flow.
- **Map or Diagram:** each view can switch from the map to a schematic
  single-line diagram (a custom component, inline HTML/CSS/JS, offline):
  substations as tiles, branches as a tidy tree, particles moving along
  each line in the real direction of power flow, lines coloured by
  loading, a gauge on each bus showing how much of its capacity the
  project uses. In the Capacity view it colours buses firm, flexible
  only or can't at a chosen size (2 MW to start).
- **Grid condition:** the Project check runs on today's grid or under
  heavy load with the other generators off, where adding solar at the far
  end visibly reverses the flow on its feeder.
- **Proof:** how well the machine-learning model matches the solver: the
  test error in one sentence and a model-versus-solver plot, with the full
  error report underneath.

## How it works

1. **The grid.** pandapower's MV Oberrhein network, generation case:
   20 kV, two 25 MVA substations, four feeders, 179 buses, 181 lines.
2. **The limits** (decisions.md): bus voltage at most 1.04 per unit, lines
   and transformers at most 100% loading.
3. **Capacity search** (`capacity.py`): a bisection search between 0 and
   20 MW, to 0.1 MW, using the real power flow at every step.
4. **ML model** (`train_model.py`): gradient-boosted trees
   (scikit-learn's HistGradientBoostingRegressor) predict the highest
   voltage, busiest line loading and busiest transformer loading for any
   bus, solar size and grid condition, in milliseconds and with no power
   flow. They were trained on 60,000 scenarios solved with the real power
   flow: 20,000 random ones, plus 40,000 placed near the pass/fail
   boundary, where accuracy matters most.
5. **95% map** (`capacity95.py`), in two stages:
   - **Model:** for 1,000 sampled conditions (load 10-60%, existing
     generation 0-80%), the model finds each bus's capacity under every
     condition; the 95% capacity is the size that fits in at least 950.
   - **Solver recheck:** the **real power flow** tests each bus's 95%
     capacity on **50 new conditions**. Wherever more than 5% fail, the
     capacity is lowered 0.1 MW and rechecked. 96 of 177 buses were
     lowered, by 0.1-0.7 MW (20.8 MW in total); none was left failing in
     more than 5% of the 50.

### How good is the model?

Tested on 12,000 scenarios it never saw:

| | Result |
|---|---|
| Pass/fail agrees with the solver | 98.0% (baseline "fits if no bigger than today's capacity": 59.3%) |
| Says pass when the solver says fail | 1.0% |
| Pass/fail on the 7,116 scenarios close to a limit | 96.6% agree |
| Highest-voltage error, close to the limit | 0.0008 pu on average |
| Capacity within 0.5 MW of the solver's search | 96.2% |

The full report is in `results/model_report.md`, and the plot in
`results/model_vs_solver.png`.

### What we found

- **Voltage is the main bottleneck.** On today's grid, voltage stops
  112 of 177 buses, a line stops 57 and a transformer stops 6.
- **Capacity falls with distance from the substation:** about 18.6 MW
  next to a substation, 4.4 MW mid-feeder and 1.9 MW at the end of the
  longest feeder (24.5 km).
- **Grid conditions matter a lot.** The far-end bus takes 1.9 MW on
  today's grid but 10.6 MW under high load and no existing solar. In 95%
  of conditions, capacities range from 3.0 MW to 20 MW or more (median
  10.0 MW, against 8.6 MW on today's grid); 173 of 177 buses have more
  room than on today's grid.

## Run it

Tested on Windows with Python 3.14.7. In PowerShell, from this folder:

```powershell
py -3.14 -m pip install pandapower==3.5.5 streamlit==1.64.0 plotly==7.1.0 scikit-learn==1.9.1 pandas==2.3.3 numpy==2.4.6 scipy==1.18.1 networkx==3.7 matplotlib==3.11.2 joblib==1.6.0 tqdm==4.70.1 pytest==9.1.1
py -3.14 train_model.py            # rebuilds the model file (about 1 minute)
py -3.14 -m streamlit run app.py   # opens http://localhost:8501
```

- The trained model file (`results/headroom_model.joblib`, 6 MB) is not in
  git, so `train_model.py` is needed once. Everything else the app shows
  is already in `results/`.
- pandapower prints a four-line numba warning when it loads. That is
  expected: numba is not installed, and the code always runs the power
  flow with `numba=False`.
- The app works offline: no map tiles, nothing loaded from the internet,
  and Streamlit's usage statistics are switched off
  (`.streamlit/config.toml`).
- Demo safety: everything the demo shows is loaded when the app first
  opens. The demo bus (147) has every slider value under both grid
  conditions precomputed in `results/demo_cache.joblib`
  (`py -3.14 precompute_demo.py`, 1 minute), so the demo never waits on
  the solver; other buses are solved live. The **Reset demo** button in
  the sidebar returns to bus 147, 0 MW, today's grid.

Tests: `py -3.14 -m pytest` (9 tests).

### Rebuilding the results from scratch

Times are on a 4-core laptop. The scenario scripts use 4 worker processes
by default (`--workers N` to change; 8 ran this laptop out of memory).

| Order | Command | Makes | Time |
|---|---|---|---|
| 1 | `py -3.14 run_capacity.py` | `results/capacity.csv`, today's capacity per bus | 2 min |
| 2 | `py -3.14 make_scenarios.py` | `results/scenarios.csv`, 20,000 random scenarios | 9-15 min |
| 3 | `py -3.14 build_features.py` | `results/bus_features.csv` | seconds |
| 4 | `py -3.14 train_model.py` | first model | 1 min |
| 5 | `py -3.14 make_scenarios.py --near-boundary` | `results/scenarios_boundary.csv` | 10-15 min |
| 6 | `py -3.14 train_model.py` | better model | 1 min |
| 7 | `py -3.14 make_scenarios.py --edges` | `results/scenarios_edges.csv` | 15 min |
| 8 | `py -3.14 train_model.py` | final model and report | 1.5 min |
| 9 | `py -3.14 capacity95.py` | `results/capacity95.csv`, the 95% map from the model | 1.5 min |
| 10 | `py -3.14 capacity95.py --recheck` | solver-rechecked `capacity95.csv` and `capacity95_recheck.json` | 14 min |

Steps 5 and 7 use the current model to choose where to sample, so their
results depend on the order above.

## Files

| File | What it does |
|---|---|
| `app.py` | the Streamlit app |
| `engine.py` | loads the grid, adds a project, solves, checks the limits |
| `capacity.py` | `find_capacity`: bisection search for one bus's capacity |
| `network_plot.py` | the offline 2D grid maps (Plotly) |
| `capacity_data.py` | reads the capacity results for the app |
| `ml.py` | the model's inputs, predictions and model-based capacity |
| `build_features.py` | fixed facts per bus: distance, impedance, line ratings |
| `make_scenarios.py` | solves training scenarios on several cores |
| `train_model.py` | trains and tests the model; report and plot |
| `capacity95.py` | the 95% capacity map with solver recheck |
| `run_capacity.py`, `run_scenarios.py` | today's capacity per bus; capacity per bus under fixed conditions (paused at 4 of 44 conditions, used as a test set) |
| `test_engine.py`, `test_capacity.py` | tests |
| `decisions.md`, `PLAN.md` | project decisions and build plan |

## Assumptions and limits

- Only an **upper** voltage limit (1.04 pu) is checked. Solar only raises
  voltage, so a lower limit cannot be what stops a project; under heavy
  load, some buses sit near 0.93 pu before any solar is added.
- One project at a time. Existing generation and load are scaled uniformly
  across the grid (decisions.md ranges); real conditions vary bus by bus
  and hour by hour.
- "95% of conditions" means 95% of conditions sampled uniformly over
  those ranges, not 95% of the hours in a real year. The solver recheck
  uses 50 conditions, so its pass rate is itself an estimate: a bus that
  truly fails in 5% of conditions will show 3 or more failures in 50
  about half the time, which is why many buses were lowered by
  0.1-0.2 MW. The lowering only ever moves capacities in the safe
  direction.
- The capacity search assumes that once a size fails, every bigger size
  fails too. The tests check that each capacity passes and that 0.1 MW
  more fails.
- The network has no short-circuit data. For the equivalent-impedance
  features, a copy of the grid assumes a 1000 MVA, R/X 0.1 upstream grid
  and generator factor k = 1.2 (decisions.md). These affect only model
  inputs, never the pass/fail results.
- The model is least accurate at the low-load, high-generation corner
  (today's grid): at the far-end bus it estimates 2.2 MW where the solver
  finds 1.9 MW. This is why the 95% map is rechecked with the solver.

## AI disclosure

This project was built with substantial help from **Claude Code**
(Anthropic's coding assistant; Claude Opus and Sonnet models), working
in this repository through prompts and reviewing its output.

**Done by Bassim Kiani:**
- The idea, the project decisions (`decisions.md`: network, limits,
  ranges, confidence level, headline size, precision) and the build plan
  (`PLAN.md`).
- Every prompt that directed the work, including what each feature should
  do and when it was done, and approval of the ML design and its changes.
- Step 1 of `find_capacity` (setting up the search range).

**Done by Claude Code:**
- Almost all of the code: `engine.py`, `app.py`, `network_plot.py`,
  `capacity_data.py`, steps 2-4 of `find_capacity` (written at Bassim's
  request, after Bassim started it), the scenario, feature, training and
  95% scripts, and the tests.
- Choices within those prompts, such as the shared "% of limit used"
  colour scale, the near-boundary training data and the solver recheck
  loop, and the analysis behind them (for example, the spot checks and
  the error breakdowns).
- Drafting this README.

Every number in this README and in the app comes from running the code in
this repository; none was written in by hand.
