# Early Phase-1 Recombination
Replacing Phase-1 bottom-slot Gaussian mutations with a mean-crossover of `pool[0]` and `pool[1]` to inject structurally blended candidates into the early descent.

## How it was tried
- stellar_p2-s11-38566380 c0016 (`-inf`): Crashed on broadcast `(2,3) vs (3,5)` because pool mixed mp=1/mp=2 seeds.
- stellar_p2-s11-38566380 c0016f (-0.600): Repaired c0016 via `_project_like` (embedding overlapping rows/cols into the target shape) and randomized the blend weight. Survived but regressed.
- stellar_p2-s17-78763752 c0001 (train 0.6188 tie, val -0.9656): Attempted high-mode recombination of top-2 bank seeds (davidkh and phanerozoic) via `_project_like` into a (12,12) canvas. Train score tied exactly, proving the vlf simulator didn't feel the perturbation. Val score collapsed (-0.9656) because the blended boundary was deeply infeasible at tighter tolerance.

## Why it failed
While structurally sound once padded, projecting out-of-shape matrices alters the strict mode-continuation logic. Linearly blending distinct local optima (especially high-mode bank seeds) introduces interference patterns in the high modes that violate physics geometry (aspect ratio, QI, elongation) while remaining invisible to loose-tolerance train sims.

## Verdict
exhausted — do not blend or interpolate across distinct pool members or bank seeds. The shape projection required to avoid crashes destroys the geometry, and the blended physics constraints destroy val feasibility.
