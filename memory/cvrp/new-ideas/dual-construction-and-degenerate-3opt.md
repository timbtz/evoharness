# Dual construction seeds & removing degenerate 3-opt

Untried (died in parse/truncation) ideas targeting initial solution variance and overlay time savings.

## How it was tried
- c0004 (cvrp-s23-89572315): Attempted polar-angle sweep construction alternative. Solver builds savings and sweep, seeds the C kernel from whichever yields lower cost. Died to SyntaxError (truncated the end of the file `_valid(c`).
- c0005r1 (cvrp-s23-89572315): Attempted to remove `_py_or_opt1`'s broken `try_intra_3opt` (which evaluates a degenerate gain identical to 2-opt, wasting time). Died to SyntaxError (accidentally deleted the entire C kernel string).
- c0006 (cvrp-s23-89572315): Successfully removed `try_intra_3opt` from the Python overlay. Train -0.117, val -0.228. Accepted as a noise-band tie. Confirms 3-opt removal is safe but yields no measurable train gradient.
- c0019/c0024 (cvrp-s23-89572315): Added `_balanced_sweep_savings_routes` (blended CW savings with angular penalties). Train -0.138 / -0.145, val -0.218 / -0.478. Rejected as noise ties.

## Why it failed / has not worked yet
- For c0004: The C kernel's ruin-and-recreate SA is highly effective at escaping the structural bias of the savings construction. Whether a structurally different sweep seed survives the first ruin pass to diversify basins is unknown.
- For c0005r1 / c0006: `try_intra_3opt` was indeed degenerate code, but its execution time was negligible. Removing it recovers milliseconds, which vanish into the eval noise band.
- For c0019/c0024: Sweep-angle balancing works correctly, but yields no immediate sprint train gradient since the SA absorbs the structural differences.

## Verdict
promising. c0004 (dual construction) is worth retrying with a properly truncated diff to see if it reliably alters the X-n200 basin landscape (addressing c0020's 0.49-0.87 private variance). c0005r1/c0006's core bugfix is factually correct but exhausted algorithmically.
