# B1/B3/B4 m=1-Selective Differential Contraction
Applying m=1-selective or m=1-row-only differential contraction (targeting the aspect-ratio dominant mode while leaving high-m QI balancing modes untouched) fails to break the aspect/QI tradeoff limit and silently regresses from the incumbent.
## How it was tried
- `stellar_p2-s105-72881323` c0051 (train 0.6241): Applied an m=1-mode-selective contraction to the B3 basin (m=1 rows received a shallower contraction factor than m≥2 rows).
- `stellar_p2-s105-72881323` c0054 (train 0.6250): Applied an m=1-row-selective contraction only to the `r_cos` matrix on B3, B4, and other bank seeds at deeper factors (-6.0e-3 to -10.0e-3).
## Why it failed
Both variants failed to improve over the exact c0029 winner (train 0.6250, val 0.6365), simply returning the incumbent floor or regressing. The m=1 mode dominates the poloidal shaping and aspect ratio; directly manipulating its amplitude disrupts the baseline geometry much more severely than uniform differential scaling, yielding no net Pareto-superior aspect/QI tradeoff.
## Verdict
exhausted — Stop isolating perturbations to the m=1 row or trying to differentially scale it. The c0009 R/Z-split uniform `base*(1+curv*(m-1))` profile remains the strict optimum.
