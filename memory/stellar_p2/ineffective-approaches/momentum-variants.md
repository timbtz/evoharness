# Complex Momentum and History Probes
Adding secondary probe directions to the momentum line search—such as ranked history vectors, orthogonal projections, cross-history recombinations, or opposing-sign quadrant probes—consistently fail or deadlock.

## How it was tried
- **Momentum-projected Gaussian (stellar_p2-s100-89908732 c0002):** Replaced standard isotropic mutations with momentum-projected Gaussian steps. Kept the parallel slope but amplified the perpendicular residual. Regressed train score to 0.5737.
- **Ranked-momentum history (stellar_p2-s100-89908732 c0004):** Kept a short history of successful step vectors. When primary momentum stalled, fired a batch probing the 2nd-best stale direction and a cross-history recombination. Deadlocked: re-evaluated the exact same two boundaries every stalled iteration (since `best_b` and `history` were frozen), burning the whole budget. Tied at 0.5688.
- **Anticipation Move (`stellar_p2-s101-20239089` c0005 `-inf`, c0005r1 val -0.015):** Speculatively batched the NEXT iteration's 1.5x momentum probe into the CURRENT `eval_many`, claiming free throughput. The writer version crashed; the refiner version drifted the boundary back inside the 1e-3 ball via un-frozen deltas and returned it through a raw-`fm.score` LF gate. The claimed throughput gain was false (the probe consumed a sequential eval slot anyway).
- **Batched Quadrant Probe (`stellar_p2-s101-20239089` c0009, c0010):** Replaced 1.5x/2.5x momentum extrapolation with batched ±sigma perturbations along the top-magnitude axis of the momentum direction. Dead decoration: tied bit-exactly at 0.6161. The probe was gated on `momentum is not None`, which never armed because the vlf-blindness death spiral meant no primary mutation was ever accepted.
- **Momentum on Escaped Boundaries (`stellar_p2-s102-48117936` c0011, c0011f):** Added 1.5x/2.5x momentum to the B3 escaped boundary. The momentum successfully found an L-gradient, but it pointed straight back into the bank seed (the homotopy trap), dragging `best_b` back inside the 1e-3 ball. Adding a hard novelty floor on acceptance (c0011f) only rejected all outward moves, starving the loop and regressing.

## Why it failed
In razor-thin feasible basins, secondary historical directions are often already exhausted or violate constraints. Re-probing them deterministically wastes budget without yielding new accepted moves. Projected Gaussians disrupt the proven isotropic + momentum line-search balance, leading to lateral regressions. Quadrant probes fail because they inherit the root polish invisibility bug—if standard momentum never arms, quadrant probes never fire. On structural escapes, the steepest L-gradient often leads straight back into the bank basin.

## Verdict
exhausted — Stick to the proven sequential 1.5x/2.5x batched momentum line search. Do not add history caches, projection matrices, or quadrant probes to the mutation logic.
