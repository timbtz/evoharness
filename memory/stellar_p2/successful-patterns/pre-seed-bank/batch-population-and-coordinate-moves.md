# Batched population + coordinate-descent moves beat plain greedy Gaussian
> HISTORICAL — pre-seed-bank program (runs s42/s7/s11: evolving the from-scratch NAE optimizer, shaped scores negative). Retired 2026-07-24 for the bank-aware seed; see performance-analysis/seed-bank-regime.md. Constants and verdicts refer to the retired trajectory.

The first three accepted mutations of the smoke run each restructured HOW the budget is spent, not the math of the step.

## How it was tried (run stellar_p2-s42-27282945, seed -0.665)
- c0001 ACC -0.564: small population (pool of parents) driven through fm.eval_many batches instead of one incumbent + sequential pairs.
- c0003 ACC -0.496: axis-aligned coordinate descent — perturb one feature/coefficient group at a time instead of full-matrix Gaussian noise.
- c0007 ACC -0.491 (smoke best): batched "scoring keys" + mirrored perturbations from the best-vs-second-best difference vector, phase structure (seed sweep -> explore -> feature search -> QI repair -> exploit), safety-floor early return.
- c0009 ACC -0.491 (run stellar_p2-s7-30186401): Deflation to coarse mode. Added `_deflate(b)` that truncates elite boundaries back to (mp=1) space before coordinate descent, then re-inflates. (Gain over c0007: 2.2e-5 — the s7 plateau.)
- stellar_p2-s7-30186401 c0014/c0017/c0027/c0028 ACC -0.491: deflation-breadth, seed-diversity and inflation-trigger tweaks, all exact ties (see implementation-insights/deterministic-plateau-and-decorations.md).
- stellar_p2-s11-38566380 c0025 ACC -0.43925: Replaced 7 baseline NAE seed specs with a "mirror-crushing" family targeting high `nfp=5` and high `aspect_ratio=10.0`, within the unchanged `SEED_BUDGET=16`.
- stellar_p2-s11-38566380 c0026f ACC -0.4392: swapped a byte-duplicate seed spec (slot 13) for an extreme `(6, 1, 10.0)` seed at zero RNG/budget cost — worst case is a guaranteed tie, best case shifts the early basin. The only two portfolio changes that ever won.

## Why it worked
72 evals is tiny; structured moves (coordinates, mirrored pairs, difference vectors, coarse-deflation) extract more signal per eval than isotropic noise, and batches keep the 2 workers busy. Dimensionality reduction guarantees complete axis sweeps in the tight budget. Swapping the seed portfolio at identical budget sizes (`c0025`) shifted early basin selection via extreme, specific geometries.

## Verdict
historical — the from-scratch NAE trajectory this tuned is retired. The transferable lessons: batch to keep workers busy, prefer structured moves over isotropic noise, and exploit zero-RNG/zero-budget swaps whose worst case is a tie.
