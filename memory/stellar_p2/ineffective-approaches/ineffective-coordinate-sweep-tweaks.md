# Ineffective Coordinate Sweep Tweaks
Reordering or deepening the coordinate sweep fails to overcome the hard plateau of the deterministic trajectory.

## How it was tried
- stellar_p2-s11-38566380 c0004 (-0.4394 plateau, accepted): Added `_deflate_min` to truncate to (1,1) single-polarity subspace. Proved to be dead code because the Phase-2 elite was already an mp=1 boundary, returning None.
- stellar_p2-s11-38566380 c0006 (-0.467) & c0006f (-0.467): Rank-prioritized coordinate sweep (largest coefficients probed first). The code reshuffled coordinates every iteration under the `coord_j` pointer, causing the sweep to re-probe the same large modes repeatedly instead of advancing. The sweep never completed an axis pass, leaving aspect ratio blown to 10.85.

## Why it failed
The hardcoded matrix-index order is naturally stable across loop iterations. Sorting by coefficient magnitude breaks this stability, and because the base boundary mutates each step, a dynamic sort degenerates into hammering the top-ranked coefficient. 

## Verdict
exhausted — keep the sweep in standard matrix-index order. If ranking coefficients, you must freeze the list for a full sweep pass.
