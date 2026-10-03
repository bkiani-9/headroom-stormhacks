from engine import add_project, check

# The search range, matching the app's solar slider.
MAX_SEARCH_MW = 20.0
PRECISION_MW = 0.1  # from decisions.md


def find_capacity(net, bus):
    """Find the largest solar size at `bus` that still passes check().

    Uses a bisection search between 0 and MAX_SEARCH_MW, to PRECISION_MW.

    Returns a pair (capacity_mw, stopping_limit):
      capacity_mw    - the largest size that passes, to 0.1 MW
      stopping_limit - the problems list from check() for the smallest size
                       that fails: the limit that stops a bigger project.
                       None if even MAX_SEARCH_MW passes.

    `net` is not changed: add_project() works on a copy.
    """
    # Step 1: Set up the search range.
    #   "low" is a size you know passes, "high" a size you know fails.
    #   Start with low = 0 MW and high = MAX_SEARCH_MW.
    #   Tip: counting in whole tenths of a MW (0 to 200) instead of MW keeps
    #   every size you test exactly on the 0.1 MW grid.
    low = 0
    high = 200

    # Step 2: Check the edges before searching.
    #   If 0 MW already fails, the grid breaks a limit with no project at all:
    #   the capacity is 0, and the limit is whatever check() reported.
    #   If MAX_SEARCH_MW passes, the capacity is at least that much:
    #   return it, with no stopping limit.
    result = check(add_project(net, bus, low / 10))
    if not result["passes"]:
        return 0.0, result["problems"]

    result = check(add_project(net, bus, high / 10))
    if result["passes"]:
        return high / 10, None
    limit = result["problems"]  # high fails: keep what stops it

    # Step 3: Repeat while low and high are more than 0.1 MW apart:
    #   a. Pick the size halfway between low and high.
    #   b. Add that much solar at `bus` and check it.
    #   c. If it passes, it becomes the new low.
    #      If it fails, it becomes the new high, and remember its problems.
    #   ("did not solve" counts as a fail.)
    while high - low > 1:  # 1 tenth = 0.1 MW
        mid = (low + high) // 2  # // divides and drops the remainder
        result = check(add_project(net, bus, mid / 10))
        if result["passes"]:
            low = mid
        else:
            high = mid
            limit = result["problems"]

    # Step 4: Return low (the largest size known to pass) and the problems
    #   remembered from high (the limit that stops it).
    return low / 10, limit
