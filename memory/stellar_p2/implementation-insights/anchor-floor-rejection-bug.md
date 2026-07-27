# The Anchor-Floor Rejection Bug (`base_b` & Diagonal Fallback Misapplication)
In branch B1, candidates that hardcoded `base_b`, raw boundaries, or the 2D diagonal as the fallback threw away the proven off-diagonal contracted incumbent, causing silent regressions.
## How it was tried
- `stellar_p2-s103-473410` c0092-c0096 (train 0.619151): Hardcoded fallbacks to raw B1_PRIMARY threw away contraction gains when search mechanisms failed gates.
- `stellar_p2-s105-72881323` c0006, c0007 (train ~0.621): Fallback chain returned `base_b` (raw uncontracted blend) instead of c0099 incumbent, causing silent regressions.
- `stellar_p2-s105-72881323` c0010-c0017 (train ~0.621): Assumed the (r_curv=z_curv) diagonal was a safe non-regressing floor. It is NOT: c0009's 0.6243 winner was an OFF-DIAGONAL split. Falling back to the diagonal silently throws away ~0.002 train score.
## Why it failed
Writers assumed `base_b`, B1_PRIMARY, or the diagonal center of a grid was a safe floor, forgetting that the accepted incumbent includes an active aspect-relief contraction that explicitly raised L.
## Verdict
recurring pitfall — NEVER hardcode the raw parent boundary, uncontracted `base_b`, or the grid diagonal as the fallback. Always set the fallback anchor to the exact off-diagonal incumbent winner (or physically reconstruct it) to guarantee non-regression.
