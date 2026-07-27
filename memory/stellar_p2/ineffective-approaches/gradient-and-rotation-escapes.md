# Random-Subspace Gradients, Phase Rotations, and Local Ascent
One-off structural perturbations (random gradients, phase rotations, additive mode deltas) fail to yield Pareto-superior QI/L tradeoffs and simply regress to the fallback floor.
## How it was tried
- `stellar_p2-s105-72881323` c0063 (train 0.6211): Random-subspace gradient line-search touching 5-10 coefficients simultaneously. Failed to beat floor, regressed to B1 anchor.
- `stellar_p2-s105-72881323` c0067 (train 0.6211): Additive curvature perturbation (delta ≈ 3e-4) on m=1, n=-1/-2 off-diagonal modes. Failed to clear feasibility/Novelty gates.
- `stellar_p2-s105-72881323` c0068 (train 0.6237): Differential m×n rotation in (R,Z) space. Preserved coefficient magnitudes but failed to produce an out-of-ball feasible improvement.
- `stellar_p2-s105-72881323` c0069 (train 0.6237): Post-selection local ascent via gradient-estimated coordinate scaling. Failed to beat incumbent.
- `stellar_p2-s105-72881323` c0070 (train 0.6237): Toroidal inflation continuation (ntor 7→8) with zero-initialized new columns. Regressed to floor.
## Why it failed
The writers predicted that changing the Fourier coefficients via orthogonal mathematical transformations (rotation, coordinated gradients, or additive deltas) without altering magnitudes would preserve QI balance while earning scale-normalized novelty distance. The code did exactly this. However, these candidate points are either strictly infeasible (failing viol/qi gates) or map to a lower shaped score than the proven tri-basin contraction incumbent. The geometry is highly sensitive, and undirected perturbations (even if mathematically orthogonal to contraction) degrade the L/QI balance.
## Verdict
exhausted — Stop trying clever mathematical perturbations (rotations, gradients, additive deltas, inflation) to escape the novelty ball. The physics will not allow it without severely degrading the score.
