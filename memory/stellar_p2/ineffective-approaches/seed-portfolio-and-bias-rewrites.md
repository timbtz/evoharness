# Seed portfolio and constraint-bias rewrites
Varying requested NAE parameters (mirror_ratio, swapping seed specs), injecting unmetered seeds, or biasing pool ranking via raw constraint heuristics. All attempts fail to beat the deterministic trajectory.

## How it was tried
- stellar_p2-s11-38566380 c0013 (-0.648): Swapped high-AR seeds to request `mirror_ratio=0.18`.
- stellar_p2-s11-38566380 c0015 (-0.482): Interleaved `mirror_ratio=0.12, 0.15, 0.20` across the portfolio.
- stellar_p2-s11-38566380 c0018 (-0.501): Swapped a single seed at index 12 for `nfp=2, mp=2, aspect=6.0`.
- stellar_p2-s11-38566380 c0003 (-0.627): Tracked constraint violations with roulettes.
- stellar_p2-s11-38566380 c0020 (-0.469): Added ALM-style mirror violation penalty to pool ranking tiebreaks.
- stellar_p2-s11-38566380 c0017 (ACC, tie): Injected 8 unmetered NAE seeds via deep copy with pseudo-scores.
- stellar_p2-s11-38566380 c0018f (-0.462): ALM penalty on secondary violations in pool ranking.
- stellar_p2-s11-38566380 c0022f (-0.4440): Injected 1 ellipsoid-perturbed seed into the unmetered sweep.

## Why it failed
Requested NAE seed parameters (mirror, iota) are NOT faithfully reproduced by mp=1 truncation. Swapping them pushes the search into fundamentally worse basins. Unmetered seed injection (`c0017`, `c0022f`) is inert because the pool truncates immediately if full, or pseudo-scores fail to overcome real evaluated elites. Heuristic pool ranking biases (`c0020`, `c0018f`) fundamentally misalign with the `-max_violation` descent signal and misread raw metrics as normalized violations, crushing geometry.

## Verdict
refuted — Never change the 16-seed NAE portfolio, never mix ellipse geometries, and never bias pool selection with hardcoded physics penalties.
