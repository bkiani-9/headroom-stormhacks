import pytest

from capacity import MAX_SEARCH_MW, find_capacity
from engine import add_project, check, farthest_bus, load_grid


@pytest.fixture(scope="module")
def net():
    return load_grid()


def is_on_tenth_grid(mw):
    return abs(mw * 10 - round(mw * 10)) < 1e-9


# 147: farthest bus, stopped by voltage. 39: at a substation, stopped by
# the transformer.
@pytest.mark.parametrize("bus", [147, 39])
def test_capacity_passes_and_next_tenth_fails(net, bus):
    assert farthest_bus(net) == 147  # the grid hasn't changed under us

    capacity, limit = find_capacity(net, bus)
    assert 0 < capacity < MAX_SEARCH_MW
    assert is_on_tenth_grid(capacity)

    at_capacity = check(add_project(net, bus, capacity))
    assert at_capacity["passes"]

    one_more = check(add_project(net, bus, round(capacity + 0.1, 1)))
    assert not one_more["passes"]
    # The limit reported is the one that stops the next size up.
    assert limit == one_more["problems"]


def test_leaves_the_grid_unchanged(net):
    sgens_before = len(net.sgen)
    find_capacity(net, 147)
    assert len(net.sgen) == sgens_before


def test_bus_that_never_fails_within_range(net):
    # Bus 71 still passes at 20 MW, the top of the search range.
    capacity, limit = find_capacity(net, 71)
    assert capacity == MAX_SEARCH_MW
    assert limit is None


def test_grid_that_fails_with_no_project():
    # Raising the substation voltage makes the grid break the voltage
    # limit before any solar is added.
    hot = load_grid()
    hot.ext_grid["vm_pu"] = 1.05
    capacity, limit = find_capacity(hot, 147)
    assert capacity == 0
    assert limit == check(hot)["problems"]
