import pytest

from engine import add_project, check, farthest_bus, load_grid


# Load the grid once for all tests. add_project returns copies,
# so the tests can't change it for each other.
@pytest.fixture(scope="module")
def net():
    return load_grid()


@pytest.fixture(scope="module")
def far_bus(net):
    return farthest_bus(net)


def test_base_case_passes():
    result = check(load_grid())
    assert result["passes"]
    assert result["problems"] == []
    assert result["max_voltage"] == pytest.approx(1.0232, abs=1e-4)


def test_zero_mw_changes_nothing(net, far_bus):
    sgens_before = len(net.sgen)
    base = check(load_grid())
    with_zero = check(add_project(net, far_bus, 0))

    assert len(net.sgen) == sgens_before  # original not touched
    assert with_zero["max_voltage"] == pytest.approx(base["max_voltage"])
    assert with_zero["max_line"] == pytest.approx(base["max_line"])
    assert with_zero["max_trafo"] == pytest.approx(base["max_trafo"])


def test_20_mw_at_far_bus_fails(net, far_bus):
    sgens_before = len(net.sgen)
    result = check(add_project(net, far_bus, 20))

    assert not result["passes"]
    assert result["problems"] != ["did not solve"]
    assert len(net.sgen) == sgens_before  # original not touched


def test_300_mw_at_far_bus_fails_without_crashing(net, far_bus):
    result = check(add_project(net, far_bus, 300))

    assert not result["passes"]
    assert result["problems"] == ["did not solve"]
