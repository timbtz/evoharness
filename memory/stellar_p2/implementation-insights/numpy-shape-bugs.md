# Variable-shape matrix bugs kill candidates
Boundary matrices have DIFFERENT shapes per seed and after inflation ((2,3) at mp=1, (3,5) after one pad, (2,5) for mp=1/ntor=2 seeds). Any blending, crossover, or stacking assumes one global shape and will crash if not projected or padded to a common canvas first.

## How it was tried
- stellar_p2-s42-27282945 (smoke): 4/15 candidates died writing population/ES code that stacked unequal shapes.
- stellar_p2-s11-38566380 c0016 (`-inf`): Phase-1 mean-crossover `_recombine_mean` failed broadcast `(2,3) vs (3,5)` because pool mixed mp=1 and mp=2 seeds.
- stellar_p2-s11-38566380 c0016f (-0.600): Fixed c0016 by projecting the partner's coarse modes into pool[0]'s canvas via `_project_like` before blending.
- stellar_p2-s11-38566380 c0024f (`-inf`): Phase-1 bracketing line-search unpacked `pool[ei]` with the assumption `(boundary, score)` instead of `(score, boundary)`, passing a `float` to `_mutate` instead of a matrix dictionary.
- stellar_p2-s11-38566380 c0026 (`-inf`): Tried to implement `_interpolate` across worst/best pool members but crashed during matrix unpacking/operations.

## Why it failed
The search pool maintains diverse mode dimensions. Arithmetic across two different members (`rc0 + w*(rc1 - rc0)`) assumes matching row/col counts. Additionally, developers frequently forget the exact tuple ordering of the internal pool structure, leading to TypeErrors during unpacking when swap attempts occur.

## Verdict
recurring pitfall — Always validate `shape` before arithmetic. Pad-to-common-canvas or use a projection slice (`rc1[:rows, ntor_t-dn:ntor_t+dn+1]`) to align the source onto the target's shape safely. Double-check tuple unpacking order against the actual `_add_candidate` implementation.
