# Toroidal-Axis (|n|-Indexed), Combined (m,n), m-Gated n-Contraction, and Joint n-axis×m-transfer Grids
Shifting aspect-relief from the m-axis to the |n|-axis, applying combined 2D (m,n) profiles, gating n-axis scaling to high-m modes, or jointly sweeping n-axis and m-transfer perturbations fails to beat the proven m-indexed R/Z split. It disrupts QI balance without unlocking new aspect margin.

## How it was tried
- `stellar_p2-s105-26196944` c0014 (REJ, train 0.6226): Reoriented the contraction profile to `factor = 1 + base*(1+ncurv*|n|/ntor)`, applying it uniformly across all rows. Also tested a tensor product `(1+mcurv*m)*(1+ncurv*|n|/ntor)`. Regressed.
- `stellar_p2-s105-26196944` c0011 (REJ, train 0.6183): Added independent scale factors (0.985–1.005) to toroidal columns `n=±1..±2`. Regressed heavily.
- `stellar_p2-s105-26196944` c0021 (REJ, train 0.6233): Applied a smooth q-differential weight `exp(-0.2·|n|)`. Regressed.
- `stellar_p2-s105-26196944` c0025 (ERR): Attempted a gentle `1+b*(1-t*|n|/ntor)` toroidal taper. Crashed (SyntaxError in matrix diff).
- `stellar_p2-s105-26196944` c0030 (REJ, train 0.6256): Damped subdominant high-|n| harmonics (m>=3, |n|>=2) by `(1-depth)`. Regressed.
- `stellar_p2-s105-26196944` c0035 (ERR, timeout 720s): Attempted a 16-eval joint m×n grid applying `[1 + bn·(1 + cn·(|n|/n_max)^2)]` post-composition. Timed out due to high eval cost on mpol=7 boundaries.
- `stellar_p2-s105-26196944` c0037r1 (ACC, train 0.6268, val 0.6378): Swept tiny n-axis scaling `[1 + tb*(n/ntor)^2]` (`tb ∈ ±2e-3`) post-composition. Scored identically to the c0034 floor, confirming toroidal scaling provides no L/aspect benefit.
- `stellar_p2-s105-26196944` c0042 (ACC, train 0.6268, val 0.6378): Applied n-axis scaling `[1 + tb*(n/ntor)^2]` ONLY to m>=2 rows to protect QI-critical m=0/m=1 modes. Scored identically to c0034 floor, proving that isolating n-axis effects cannot unlock aspect relief.
- `stellar_p2-s105-26196944` c0045f (REJ, train 0.6259, val 0.6373): Predicted that a joint 2D sweep of m-gated n-axis scaling (`tb`) and low-to-high m curvature transfer (`t`) would find a Pareto-superior point that single-lever sweeps missed. The code combined them in a `_combine()` function applying both multipliers sequentially. Regressed. The joint cross-terms simply compound the disruption to QI balance.

## Why it failed
The writers predicted that perturbations along the toroidal n-axis (continuous n-axis scaling, selective damping, m-gating, or joint grids with m-transfers) would decouple aspect-relief from QI-critical harmonics or find hidden Pareto improvements. However, independently manipulating toroidal harmonics disrupts the delicate magnetic surface QI balance without unlocking any new aspect relief. It yields no net Pareto-superior aspect/QI tradeoff and simply breaks the proven geometry of the winning m-indexed split.

## Verdict
exhausted — Stop trying independent toroidal-mode (n-axis), combined (m,n), m-gated n-scaling, per-column differential scaling, `exp(-q·|n|)` weighting, or joint n-axis/m-transfer grids. The single off-diagonal R/Z split at `(cr,cz)=(0.5, 0.7)` remains a strict optimum.
