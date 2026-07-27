# B1 Power-Law, Fabricated Boundaries, and Additive High-M Band Contraction
Steepening high-m contraction via nonlinear profiles, fabricating VMEC-convergent matrices from scratch, or additive bands fails to provide a Pareto-superior aspect-relief/QI tradeoff compared to linear scaling, and risks solver crashes.
## How it was tried
- `stellar_p2-s105-72881323` c0005f (train 0.6221): Swept power-law exponents (1.3, 1.7, 2.0). Plateaued slightly above linear baseline.
- `stellar_p2-s105-72881323` c0015f (train 0.6214): Applied an additive high-m band term `hi·max(0,m-3)` to target localized modes setting L. Failed to beat the exact linear anchor.
- `stellar_p2-s105-26196944` c0002 (ERR): Generated a fabricated low-mode-count (mpol=4, ntor=2) NAE-derived nfp=3 escape boundary and applied an R/Z-split sweep. Crashed VMEC with `FATAL ERROR... initial boundary is poorly shaped`.
- `stellar_p2-s105-26196944` c0003 (ERR): Hardcoded another manually approximated B3 boundary with only 5 rows and 9 cols. Crashed the harness with `r_cos for m=0 and n<0 must be 0.0`.
- `stellar_p2-s105-26196944` c0005f (ERR): Hardcoded a tiny 5x3 matrix representation of the B3 escape and attempted an exponential `(m-1)^0.8` deepening. Crashed VMEC with `Minimum of B is at the boundary`.
## Why it failed
Altering the contraction distribution to target high-m modes specifically disrupts the mid-m QI geometry without unlocking new aspect margin. The grid search has fully saturated the profile shape space. Furthermore, manually approximating or truncating B3-lhhhhappy3 into a short hardcoded string destroys the delicate spectral condensed property required by VMEC to initialize, leading to immediate `RuntimeError` crashes.
## Verdict
exhausted — Stick to the exact linear differential scaling profiles on a verified VMEC-safe matrix. Hand-typing approximations of bank seeds or low-resolution NAE templates is a strict trap.
