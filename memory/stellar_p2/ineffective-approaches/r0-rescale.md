# Uniform Major-Radius (R0) Rescale Pre-Contraction
Applying a uniform global rescale to the major radius R0 (`rc[0, ntor] *= s`) before the proven two-stage m-differential contraction fails to unlock a Pareto-superior aspect ratio shift, regressing the score.

## How it was tried
- `stellar_p2-s105-26196944` c0044r1 (REJ, train 0.6219, val 0.6333): Swept rescale factors `[0.97, 1.01, 1.02, 1.03]` on the B3 escape boundary before applying the proven two-stage contraction.

## Why it failed
The writer predicted that scaling R0 uniformly shifts the global aspect ratio without distorting the spectral shape, creating an orthogonal aspect-relief lever. The code did exactly this. However, the base boundary's aspect ratio is already heavily constrained by its internal spectral coupling; artificially scaling the axisymmetric R0 term out of sync with the natural minor radius breaks the QI balance and induces a geometric penalty that outweighs any raw aspect ratio shift.

## Verdict
refuted — Stop applying global axisymmetric rescales to manipulate aspect ratio. The only viable aspect-relief mechanism remains the spectral m-differential R/Z split.
