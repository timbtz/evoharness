# Random-Subspace Gradients, Phase Rotations, and Local Ascent
One-off structural perturbations (random gradients, phase rotations, additive mode deltas, local ascent, and `n`-axis R/Z-splits) fail to yield Pareto-superior QI/L tradeoffs and simply regress to the fallback floor.
## How it was tried
- `stellar_p2-s105-72881323` c0063 (train 0.6211): Random-subspace gradient line-search touching 5-10 coefficients simultaneously. Failed to beat floor, regressed to B1 anchor.
- `stellar_p2-s105-72881323` c0067 (train 0.6211): Additive curvature perturbation (delta ≈ 3e-4) on m=1, n=-1/-2 off-diagonal modes. Failed to clear feasibility/Novelty gates.
- `stellar_p2-s105-72881323` c0068 (train 0.6237): Differential m×n rotation in (R,Z) space. Preserved coefficient magnitudes but failed to produce an out-of-ball feasible improvement.
- `stellar_p2-s105-72881323` c0069 (train 0.6237): Post-selection local ascent via gradient-estimated coordinate scaling. Failed to beat incumbent.
- `stellar_p2-s105-72881323` c0070 (train 0.6237): Toroidal inflation continuation (ntor 7→8) with zero-initialized new columns. Regressed to floor.
- `stellar_p2-s100-78100567` c0056 (train 0.6128): Pre-contraction spectral reweighting `exp(-α|m|)` vs `exp(-β|m|)` applied independently to `r_cos` and `z_sin`. Disrupted QI balance and regressed to floor.
- `stellar_p2-s100-78100567` c0058 (train 0.6131): Toroidal-mode-selective R/Z contraction applying a Gaussian envelope `exp(-n²/2w²)` to off-axis (`n≠0`) harmonics. Failed to beat incumbent, returning to the fallback floor.
## Why it failed
The writers predicted that changing the Fourier coefficients via orthogonal mathematical transformations (rotation, coordinated gradients, additive deltas) or applying the proven asymmetric R/Z recipe to new axes (`n` instead of `m`) would preserve QI balance while earning scale-normalized novelty distance. The code did exactly this. However, these candidate points map to a lower shaped score than the proven tri-basin contraction incumbent. The geometry is highly sensitive, and perturbations to the `n`-axis or high-`m` spectral structure degrade the L/QI balance without unlocking any new aspect relief.
## Verdict
exhausted — Stop trying clever mathematical perturbations (rotations, gradients, additive deltas, inflation, spectral envelopes) to escape the novelty ball. The physics will not allow it without severely degrading the score.
