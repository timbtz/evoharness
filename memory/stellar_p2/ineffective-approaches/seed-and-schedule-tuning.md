# Elite pool widening and sigma tuning
Broadening the elite pool for mutations (K=4 or K=5 instead of K=3) or tweaking the sigma decay factor (SHRINK=0.75 vs 0.83) yields negligible to negative returns and often breaks carefully balanced exploration dynamics.

## How it was tried
- stellar_p2-s7-30186401 c0016 (-0.712): Widened Phase 1 exploration pool from top-3 to top-5.
- stellar_p2-s7-30186401 c0025 (-0.595): Widened Phase 1 exploration pool from top-3 to top-4.
- stellar_p2-s7-30186401 c0026 (-0.571): Lowered sigma shrink factor from 0.83 to 0.75.
- stellar_p2-s7-30186401 c0021 (-0.751): Changed mutations to scale proportionally to coefficient magnitude.
- stellar_p2-s11-38566380 c0023 (-0.580): Applied 1.2x-1.3x scaling to requested `rotational_transform` to fight mp=1 truncation iota deficits.

## Why it failed
The incumbent's hardcoded constants (K=3, SHRINK=0.83, `ref*damp` scaling) are already well-tuned for the 72-eval budget. Widening the pool introduces worse-scoring elites that waste evaluations. Making sigma decay faster starves the late-game QI repair phase of the step size it needs. Altering the relative scaling of perturbations destroys the ALM-inspired spectrum scaling that keeps high modes from crashing the geometry. Rescaling the NAE `rotational_transform` doesn't map linearly to measured iota and disrupts the deterministic seed basin.

## Verdict
refuted — do not tweak global mutation knobs, elite selection widths, or seed scalar parameters. The search survival relies on these exact values.
