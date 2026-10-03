"""Load hosting-capacity results for the Capacity view.

The app only reads results from a file; it never runs the capacity search.

File format, results/capacity.csv, one row per bus:

    bus,capacity_mw,limit
    147,1.9,voltage 1.0404 pu is above 1.04
    39,19.9,HV/MV Transformer 0 is loaded to 100.4%
    12,20.0,

  bus          - the bus index (as in the app's dropdown, not its name)
  capacity_mw  - the largest solar size that passes, to 0.1 MW
  limit        - the limit that stops a bigger project, in plain words;
                 leave it empty if nothing breaks within the search range
"""
from pathlib import Path

import pandas as pd

CAPACITY_FILE = Path(__file__).parent / "results" / "capacity.csv"
HEADLINE_MW = 2.0  # headline project size, from decisions.md


def read_capacity(path=CAPACITY_FILE):
    """Read the capacity file into a table indexed by bus."""
    table = pd.read_csv(path, dtype={"limit": str}, keep_default_na=False)
    return table.set_index("bus")[["capacity_mw", "limit"]]


def count_can_take(table, mw):
    """How many buses in `table` have a capacity of at least `mw`."""
    # A tiny tolerance, so a capacity saved as 2.0 counts for a 2 MW project.
    return int((table.capacity_mw >= mw - 1e-9).sum())


def placeholder_capacity(buses):
    """Obviously fake capacities, so the view can be built before the
    real file exists. A repeating 0-20 MW sawtooth by bus number: it has
    nothing to do with the grid, on purpose.
    """
    rows = [
        {"bus": b, "capacity_mw": (b * 7 % 201) / 10, "limit": "placeholder"}
        for b in buses
    ]
    return pd.DataFrame(rows).set_index("bus")
