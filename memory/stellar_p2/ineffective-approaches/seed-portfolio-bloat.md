# Seed portfolio bloat and direct ellipse mixing
Expanding `SEED_BUDGET` to evaluate more initial seeds, or mixing in different geometry families (like `fm.seed_ellipse`), destroys the budget balance and introduces unsearchable starting points.

## How it was tried
- stellar_p2-s7-30186401 c0022 (-0.661): Mixed `fm.seed_ellipse` geometries and `mp=2` NAE seeds into the portfolio.
- stellar_p2-s7-30186401 c0023 (-0.855): Raised `SEED_BUDGET` from 16 to 22.
- stellar_p2-s7-30186401 c0024 (-0.614): Forced all NAE seeds to `mp=1` for smoother boundaries.

## Why it failed
The 72-eval budget requires a strict balance: 16 evals for seeds, 56 for optimization. Taking 6 evals away from optimization to look at more seeds destroys the downstream phase trajectories. Mixing in ellipse geometries introduces fundamentally different search spaces that the highly specific deflation/inflation logic (tuned for NAE matrices) cannot handle. Forcing all seeds to `mp=1` removes necessary mode diversity.

## Verdict
refuted — keep `SEED_BUDGET` at 16. Stick to the narrow NAE parameter combinations already established.
