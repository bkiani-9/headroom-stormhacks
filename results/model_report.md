# Headroom ML model: error report

Trained on 32,000 solved scenarios; tested on 8,000 scenarios it never saw.

## Predicting the solver's numbers (held-out scenarios)

| Quantity | Average error | 95% of errors below | Worst error |
|---|---|---|---|
| Highest voltage | 0.0009 pu | 0.0025 pu | 0.0170 pu |
| Busiest line loading | 1.03% | 3.05% | 13.47% |
| Busiest transformer loading | 0.86% | 2.24% | 10.45% |

## Pass or fail (held-out scenarios)

| | Agrees with solver | Says pass, solver fails (dangerous) | Says fail, solver passes |
|---|---|---|---|
| Model | 98.1% | 1.0% | 0.9% |
| Baseline: "fits if no bigger than today's capacity" | 64.1% | 0.0% | 35.9% |

## Capacity (against find_capacity, 708 bus-and-condition cases, 4 conditions)

- Average error: 0.28 MW; worst: 2.2 MW
- Within 0.5 MW of the solver: 85.3%
- Over-predicted by more than 0.5 MW (dangerous): 5.5%

The model is never trusted alone: step 6 rechecks every capacity with the real solver before it goes on the map.
