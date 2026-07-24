# Full schedule rewrites and unified adaptive loops
Rewriting the incumbent's phase schedule, phase boundaries, or early-loop decision trajectories consistently destroys search performance, dropping scores to the -0.62 to -0.84 range.

## How it was tried
- stellar_p2-s7-30186401 c0010 (-0.840): Ripped out phases for a nevergrad CMA-ES loop. Score collapsed.
- stellar_p2-s7-30186401 c0011 (-0.747): Batched seeds and simplified the loop into a continuous coordinate descent. Halving sequential evaluations regressed the score heavily.
- stellar_p2-s7-30186401 c0012 (-0.677): Replaced phases with an adaptive difference-vector exploitation loop.
- stellar_p2-s7-30186401 c0013r1 (-0.623): Stall-adaptive loop replacing rigid phase boundaries.
- stellar_p2-s11-38566380 c0002 (-0.646): Replaced fixed Phase-1 sequential block with adaptive `EXPLORATION_ITERS` loop, deriving `phase2_start` from new constant.
- stellar_p2-s11-38566380 c0009 (-0.523): Adaptive seed-pair bandit + early Gaussian descent replacing fixed 16-seed sweep.
- stellar_p2-s11-38566380 c0010 (-0.643): Multi-rank early descent burst (top-2 elites perturbed at sigma and 2.5*sigma simultaneously) replacing fixed-Phase-1.

## Why it failed
The success of the incumbent relies on a tightly coupled, hardcoded trajectory of exploration (Phase 1) -> feature search (Phase 2) -> QI repair (Phase 3) -> exploit (Phase 4). Attempts to make this "smarter" or "continuous" consistently strip out the highly-tuned budget allocation that fits exactly into the tight evaluation budget. 

## Verdict
exhausted — stick to surgical 1-2 line edits to the existing phase schedule. Do not restructure the optimization loop.
