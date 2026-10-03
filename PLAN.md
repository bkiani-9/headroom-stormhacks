# Headroom build plan

Hacking started 12:00 pm Saturday. Times are deadlines.

## Must-haves
- [ ] 1. Grid solves in the app and shows about 1.0232 (**H1, 1:00 pm**)
- [ ] 2. Limit check: the base case passes and 20 MW at a far bus fails; a rough version works end to end (**H3, 3:00 pm**)
- [ ] 3. Map that reacts: click a point, drag the slider, the map recolours; before/after toggle (**H8, 8:00 pm**)
- [ ] 4. My capacity search and the heatmap: hovering shows the limit that stops each point (**H11, 11:00 pm**)
- [ ] 5. ML model: trained on solver runs, with an error report and a model-versus-solver plot (**H14, 2:00 am**)
- [ ] 6. 95% capacity map: physics recheck, snapshot/95% toggle and the headline number (**H16, 4:00 am, feature freeze**)

## ML design for step 5 (APPROVED)

Goal: a model that predicts a bus's hosting capacity under any grid
condition in milliseconds, so step 6 can sample hundreds of conditions per
bus instead of running the solver hundreds of times.

### What one scenario contains
- A grid **condition**: two numbers, applied to every load and every
  existing generator on the grid.
  - load level: 10% to 60% of each load's rated size
  - existing generation level: 0% to 80% of each generator's rated size
  - These are exactly pandapower's two built-in cases at the corners:
    "load" (60%, 0%) and "generation" (10%, 80%, the case we use today).
- A **bus**: one of the 177 buses at 20 kV.
- One scenario = one condition + one bus. Every condition is run at all
  177 buses, so the grid is solved once per condition and shared.

### How conditions are sampled
- 40 conditions, spread evenly over the 10-60% x 0-80% square (Latin
  hypercube, fixed random seed so it can be rerun exactly), plus the four
  corners.
- 44 conditions x 177 buses = 7,788 scenarios. At about 0.7 s each,
  that's about 90 minutes of solver time, run in the background with a
  progress bar.

### What the solver records (one row per scenario)
- condition: load level, generation level
- bus
- capacity_mw: from find_capacity, 0 to 20 MW in 0.1 MW steps
- limit type: voltage, line, transformer, or none (20 MW+)
- the full limit text, e.g. "Line 62 is loaded to 100.3%"
- run time
- Saved to results/scenarios.csv. Saved every condition, so a crash
  loses at most one condition.

### Features the model sees
Cheap to get: fixed facts about the bus, plus one ordinary power flow
per condition (no capacity search).
- load level, generation level
- the bus's distance from its substation: resistance and reactance
  (ohms) and length (km)
- which substation feeds it
- the bus's voltage and the highest voltage on its feeder, in that
  condition with no new solar (from the one shared power flow)
- the busiest line between the bus and its substation, in that condition

Not used: the capacity itself or anything from the capacity search (that
would be the answer leaking into the question).

### Model
- Gradient-boosted trees (scikit-learn's HistGradientBoostingRegressor,
  already installed; no new libraries). Good on small tables like this,
  trains in seconds.
- Capacities of 20 MW mean "20 or more": the model predicts at most 20.

### How we test it
- Hold out whole conditions, not random rows: 34 conditions to train, 10
  never seen in training to test. (Random rows would let the model see
  the same condition at neighbouring buses, which flatters it.)
- Report on the 10 test conditions:
  - average error (MW) and worst error (MW)
  - share of predictions within 0.5 MW of the solver
  - share that **over**-predict by more than 0.5 MW (the dangerous
    direction: it would say a project fits when it doesn't)
  - the same numbers split by limit type (voltage / line / transformer)
- Compare with a simple baseline: "use today's capacity from
  results/capacity.csv, whatever the condition". The model must beat it.
- Plot: model vs solver capacity, one dot per test scenario, with the
  perfect-prediction line.
- Step 6 never trusts the model alone: every 95% capacity is rechecked
  with the solver before it goes on the map.

### Decided
1. 44 conditions (about 90 minutes of solving).
2. Load and generation move together across the whole grid: one load
   level and one generation level per condition.
3. A good prediction is within 0.5 MW of the solver.

## Stretch: only if ahead at a checkpoint, never after H16
- [ ] Power-factor toggle (inverters absorbing reactive power)
- [ ] Animate a day with SimBench load and solar profiles
- [ ] Battery placement for a failing project
- [ ] Plain-language summary that only rephrases the solver's numbers

## After the freeze: sleep, then polish
- [ ] Sleep 4:00 to 7:00 am
- [ ] Design pass: one colour scale, headline at the top, readable from a metre away
- [ ] 60-second backup video and 20-second GIF
- [ ] README and Devpost page, with the AI disclosure
- [ ] Rehearse the 3 minutes five times, timed
- [ ] Submit by 11:00 am Sunday, an hour before the deadline
