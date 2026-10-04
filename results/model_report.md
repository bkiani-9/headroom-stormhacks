# Headroom ML model: error report

Trained on 48,000 solved scenarios (80%); tested on 12,000 scenarios it never saw (20%). Data: scenarios.csv, scenarios_boundary.csv, scenarios_edges.csv.

## Predicting the solver's numbers (held-out scenarios)

| Quantity | Average error | 95% of errors below | Worst error |
|---|---|---|---|
| Highest voltage | 0.0009 pu | 0.0026 pu | 0.0115 pu |
| Busiest line loading | 0.94% | 2.73% | 20.01% |
| Busiest transformer loading | 0.82% | 2.18% | 10.39% |

## Close to a limit (held-out scenarios)

Cases where the solver's value is near a limit (voltage within 0.01 pu of 1.04; loading within 10 points of 100%), where a small error can flip pass and fail. Average bias: negative means the model guesses too low (too optimistic).

| Quantity | Cases | Average error | 95% of errors below | Worst error | Average bias |
|---|---|---|---|---|---|
| Highest voltage | 4,740 | 0.0008 pu | 0.0020 pu | 0.0054 pu | -0.0001 pu |
| Busiest line loading | 3,684 | 0.73% | 1.89% | 4.16% | -0.06% |
| Busiest transformer loading | 28 | 0.72% | 1.98% | 2.75% | -0.13% |

Pass or fail on the 7,116 scenarios close to any limit: agrees with the solver 96.6%; says pass when the solver fails 1.7%; says fail when the solver passes 1.7%.

## Pass or fail (all held-out scenarios)

| | Agrees with solver | Says pass, solver fails (dangerous) | Says fail, solver passes |
|---|---|---|---|
| Model | 98.0% | 1.0% | 1.0% |
| Baseline: "fits if no bigger than today's capacity" | 59.3% | 0.0% | 40.7% |

## Capacity (against find_capacity, 708 bus-and-condition cases, 4 conditions)

- Average error: 0.18 MW; worst: 1.8 MW
- Within 0.5 MW of the solver: 96.2%
- Over-predicted by more than 0.5 MW (dangerous): 0.8%

The model is never trusted alone: step 6 rechecks every capacity with the real solver before it goes on the map.
