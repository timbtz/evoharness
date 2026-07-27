# Cross-Basin Recombination, Mode-Grafting, Same-nfp Blends, and Homotopy
Splicing modes from different nfp basins, blending same-nfp basins (B1↔B3), or executing same-nfp convex homotopy fails to improve L and typically plateaus or crashes.
## How it was tried
- `stellar_p2-s105-72881323` c0058 (train 0.6237): Spliced B1's proven low-order modes into B4's zero-violation background. Failed to improve over the c0029 incumbent.
- `stellar_p2-s105-72881323` c0060 (ERR): Attempted inter-nfp B3↔B4 mode-grafting via per-row structural blending. Crashed with `SyntaxError: cannot use assignment expressions`.
- `stellar_p2-s105-72881323` c0062 (train 0.6237): Attempted ellipticity gradients and toroidal-phase rotations as orthogonal escape axes. Regressed to the fallback floor.
- `stellar_p2-s105-72881323` c0079 (train 0.6255, val 0.6368): Attempted a same-nfp convex homotopy blend between two nfp=4 bank seeds as a parallel escape mechanism. Regressed safely to the nfp=3 fallback floor.
- `stellar_p2-s105-72881323` c0081 (ERR): Generated convex blends between the B1 and B3 nfp=3 basins and applied a fine depth contraction sweep to each. Crashed with a `JSONDecodeError` during hardcoded string payload manipulation.
## Why it failed
Writers predicted that sharing the same nfp or interpolating basins would preserve dominant shaping harmonics while crossing the novelty ball. However, mixing basins or convexly blending them destroys the delicate magnetic surface coupling required for QI balance. For B1↔B3 blends, the code physically scaled into an asymmetric global shift that offered no new Pareto-front headroom, just confirming the existing plateau or hitting syntax/formatting landmines.
## Verdict
exhausted — Stop splicing or blending Fourier modes across basins. Same-nfp recombination and cross-nfp grafting are structurally pathological or strictly redundant.
