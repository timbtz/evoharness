# Route-set crossover overlay (dual C LNS calls)

Running two independent C LNS calls to perform route-level recombination (replacing routes in solution B with cheaper routes from solution A if they serve the same customer set).

## How it was tried
- c0012 (run cvrp-s17-78412700): Added a `_run_c_lns` wrapper to execute two C calls (different seeds), then `_route_set_crossover` using `frozenset(route)` mapping. Train -0.2409, val -0.2444. Rejected.

## Why it failed
- Time-starvation: splitting the 5s budget starves the SA cooling schedule of both calls, heavily regressing train.
- Identical route customer sets are rare across highly optimized distinct SA trajectories, yielding zero useful recombination moves.

## Verdict
refuted. Do not use multiple C LNS calls to enable route-level crossover.
