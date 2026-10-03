# Headroom: rules for Claude

Read decisions.md for the project decisions and PLAN.md for the build order.

- I'm on Windows with PowerShell, so give me commands that work in PowerShell.
- I'm new to Python, so explain in plain words.
- Run Python as `py -3.14` (pip too: `py -3.14 -m pip ...`).
- numba isn't installed, so always call `runpp(net, numba=False)`.
- One small step per prompt. After each change, run it if you can, or give me the exact command and what I should see.
- Commit after each step that works.
- I write `find_capacity` myself, so never write its logic.
- No new libraries without asking.
- The finished app must work offline: nothing loaded from the internet while it runs (no online map tiles, no scripts from a CDN).
