# Anisotropic (m,n) Exponential Escape off Top Bank Seed
Applying `exp(-a*m - b*|n|)` scaling to the top-scoring bank seed was hypothesized to reduce eval cost while structurally leaving the 1e-3 novelty ball. Instead, it consistently collapsed the score to the B4 fallback floor (~0.6237).
## How it was tried
- `stellar_p2-s105-72881323` c0055 (train 0.6240): Applied the anisotropic scaling mechanism to B3 and B4 basins. Score plateaued below c0029.
- `stellar_p2-s105-72881323` c0055f (train 0.6237): Redirected the mechanism to the top-scoring bank seed, collapsing eval cost. The scaled candidates collapsed straight to the B4 fallback floor.
- `stellar_p2-s105-72881323` c0057 (train 0.6237): Applied m≥2-only exponential high-mode damping to the top bank seed's non-axisymmetric modes. Collapsed to the B4 fallback floor.
- `stellar_p2-s105-72881323` c0060f (train 0.6237): Attempted a highly-touted timeout-guarded escape off the highest-scoring bank seed with hard-truncation to mpol=8/ntor=7. Collapsed to the B4 fallback floor.
## Why it failed
The writer predicted that exponential high-mode suppression would "collapse eval cost" while raising L by removing "grid wiggle." The code did exactly this: it applied `factor(m,n)=exp(−(a_m·m + a_n·|n|))` to the coefficients. However, the top bank seed is already L-maximized at its QI feasibility wall. Any structural damping or hard truncation of its modes disrupts the delicate spectral balance, degrading L and QI margin and dropping the candidate to the inferior fallback floor.
## Verdict
refuted — Stop applying anisotropic exponential scaling, high-mode damping, or hard truncation to the top-scoring bank seed. The top seed is structurally Pareto-blocked.
