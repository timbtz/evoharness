# Batched population + coordinate-descent moves beat plain greedy Gaussian
The first three accepted mutations of the smoke run each restructured HOW the budget is spent, not the math of the step.

## How it was tried (run stellar_p2-s42-27282945, seed -0.665)
- c0001 ACC -0.564: small population (pool of parents) driven through fm.eval_many batches instead of one incumbent + sequential pairs.
- c0003 ACC -0.496: axis-aligned coordinate descent — perturb one feature/coefficient group at a time instead of full-matrix Gaussian noise.
- c0007 ACC -0.491 (smoke best): batched "scoring keys" + mirrored perturbations from the best-vs-second-best difference vector, phase structure (seed sweep -> explore -> feature search -> QI repair -> exploit), safety-floor early return.
- c0009 ACC -0.491 (run stellar_p2-s7-30186401): Deflation to coarse mode. Added `_deflate(b)` that truncates elite boundaries back to (mp=1) space before coordinate descent, then re-inflates.
- stellar_p2-s7-30186401 c0014 ACC -0.491: Deflation intensified. `_deflate()` now truncates toroidal modes (columns) to n=±1 in addition to m=0,1 rows.
- stellar_p2-s7-30186401 c0017 ACC -0.491: Diversified NAE seed `n_field_periods` toward 5 and 4.
- stellar_p2-s7-30186401 c0027 ACC -0.491: Raised inflation stall threshold from 8 to 14.
- stellar_p2-s7-30186401 c0028 ACC -0.491: Added a second inflation step to mp=3 when stalled post-first-inflation.
- stellar_p2-s11-38566380 c0025 ACC -0.43925: Replaced 7 baseline NAE seed specs with a "mirror-crushing" family targeting high `nfp=5` and high `aspect_ratio=10.0`, within the unchanged `SEED_BUDGET=16`.

## Why it worked
72 evals is tiny; structured moves (coordinates, mirrored pairs, difference vectors, coarse-deflation) extract more signal per eval than isotropic noise, and batches keep the 2 workers busy. Dimensionality reduction guarantees complete axis sweeps in the tight budget. Surgical edits to deflation breadth, seed mixes, and inflation triggers preserve the perfect budget trajectory. Swapping the seed portfolio at identical budget sizes (`c0025`) proved that shifting early basin selection via extreme, specific geometries outperforms the diverse baseline.

## Verdict
promising — build on c0025's basin shift. The next lever is either finding further beneficial NAE parameter extremes (without budget bloat) or repairing the coordinate sweep to properly traverse newly created fine modes from inflation.
