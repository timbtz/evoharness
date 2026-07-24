# Alternating-sign coordinate sweeps
Flipping the probe sign (+/-) each time the coordinate sweep wraps to a new pass. Attempted to access negative-descent directions at zero extra budget cost.

## How it was tried
- stellar_p2-s11-38566380 c0013f (-0.4412), c0019f (-0.4412): Altered `step = sweep_sign * sigma * base * 0.8` and flipped `sweep_sign` when `coord_j` wrapped.

## Why it failed
The deflated elite search space has ~7-8 live coordinates. In the 72-eval budget, the search triggers Phase transitions and inflations long before the sweep completes one pass, let alone a second pass. The flip mechanism never fires, resulting in an exact score tie (-0.4412 vs parent -0.4394) or noise regression.

## Verdict
exhausted — If attempting to sweep both directions, you must do it inline within the same pass (e.g., bidirectional probes, which are also refuted due to budget cost) rather than relying on a future wrap.
