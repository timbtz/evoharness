# Axis-locked Coordinate, Greedy Descent, & Phase Probes in Bank Polish
Replacing the isotropic Gaussian micro-polish with axis-locked coordinate sweeps, greedy descent, or R/Z phase rotations consistently exhaust the budget, break geometry, or dead-lock.

## How it was tried
- `stellar_p2-s17-78763752` c0002 (ACC tie): ~66 coordinates, probing ±2 amps each. Exhausted budget before Phase 2.
- `stellar_p2-s17-78763752` c0003 (`-inf`): 2D rotation matrix on `r_cos`/`z_sin`. Hard timeout (destroyed cross-section geometry).
- `stellar_p2-s17-78763752` c0004 (Rej): 3% minor radius dilation. Regressed train by pushing aspect ratio to hard bound.
- `stellar_p2-s101-20239089` c0012 (timeout): Implemented low-fidelity finite-difference gradient ascent over 20-30 coefficients. Exhausted 720s CPU limit (40-60+ slow lf evals). Refuted as re-implementing a numeric optimizer core.
- `stellar_p2-s101-20239089` c0013 (ACC tie): Added gauged greedy descent with hard L-lock (`delta_L > 5e-4`). Still tied exactly at 0.6161 and risked timeout. The escape is a strict local L-optimum; visible moves just break feasibility (aspect wall).

## Why it failed
Bank seeds sit in a razor-thin feasible basin, pinned at the aspect-ratio wall. Axis-locked coordinate probes or greedy descent consume 2 evals per mode, rapidly exhausting the tight budget. Because the escape boundary is a local L-optimum against the aspect constraint, any visible directional step pushes through the constraint wall and is rejected by the L-prioritizing acceptance key. Phase rotations are unphysical in this representation, lacking `r_sin`/`z_cos` terms.

## Verdict
refuted — Do not replace the Gaussian micro-polish with coordinate sweeps, finite-difference gradients, greedy descent, R/Z phase rotations, or blunt minor-radius dilations.
