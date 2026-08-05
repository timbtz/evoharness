# Graded soft-fail + warm evals: exploit the new eval contract (2026-08-01)

Since Plan 1 (commit be6cfd4) `fm.eval`/`fm.eval_many` changed in two ways that
reward specific optimizer structure. Status: UNTESTED as deliberate strategy —
live A/B showed the machinery works (82 warm, 9 soft in one run) but only 3/19
candidates exploited it. First program to use these deliberately gets clean
attribution.

## 1. Non-converged solves now return graded data, not None

A boundary whose VMEC solve doesn't converge returns a metrics dict with
`soft_fail: True`, `p2_score 0.0`, `feasibility inf`, and a graded
`shaped_score` in [-1001, -1000] — HIGHER = closer to convergence (log10 of
force residuals vs ftol; also raw `fsqr/fsqz/fsql/niter` included).

How to exploit (recipe): treat the convergence cliff as a bracketable boundary.
If parent p converged and child c = p + step soft-fails, DON'T discard the
direction — bisect: eval p + step/2, p + step/4... The graded score tells you
which of two failing steps is nearer solvable, so you can walk INTO the
high-curvature region where L∇B gains live instead of retreating to safe
ground. Costs 1 budget unit per probe like any eval. Caveat: the sentinel only
covers slow non-convergence; hard solver errors still return None.

## 2. Near-parent evals are ~25% cheaper (warm path)

If a boundary is within max-coeff 1e-3 of a recently evaluated CONVERGED
boundary with the same nonzero-mode footprint (same grid), the solve skips a
wasted multigrid stage: ~25% less wall-clock, metrics identical (77% of pairs
bit-identical; worst p2 drift 2e-3, train-signal only). The cache remembers
the last 4 converged parents per worker; batches alternate 2 workers.

How to exploit: structure searches as LADDERS (sequences of ≤1e-3 steps from
one parent), not scattered portfolios; keep the mode footprint fixed within a
ladder (padding a matrix with a new nonzero mode changes the grid = cold);
batch ladder members together. Wall-clock is the binding budget on big-mode
boundaries (~13s/eval cold) — 25% cheaper evals = ~1/3 more evals per
candidate exactly where the (8,15)-family search is starved.
