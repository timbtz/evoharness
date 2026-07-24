# Coordinate probes and sweep rewrites
Rewriting Phase 1 Gaussian mutations with axis-aligned probes, or sorting coordinate sweeps by magnitude. These consistently destroy the tightly tuned budget trajectory.

## How it was tried
- stellar_p2-s11-38566380 c0011 (-0.570): Basin-jump transition injecting full-matrix ±sigma*4.0 probes after early descent stall.
- stellar_p2-s11-38566380 c0012 (-0.641): Replaced early Gaussian mutations with priority-ordered bidirectional axial probes.
- stellar_p2-s11-38566380 c0014 (-0.583): Replaced early mutations with batched coordinate sweeps on dominant R/Z modes.
- stellar_p2-s11-38566380 c0006 (-0.467): Rank-prioritized coordinate sweep (largest coefficients probed first).

## Why it failed
Bidirectional probes were refuted 3x (`bidirectional-coordinate-probes` wiki) because they cost 2 evals per step, halving axis coverage in the 72-eval budget. `c0012` and `c0014` repeated this exact failure. Rank-prioritized sweeps (`c0006`) dynamically reshuffle coordinates every loop, causing the sweep to re-probe the top mode repeatedly while leaving others unsearched. 

## Verdict
refuted — Keep the sweep in standard matrix-index order. Do not batch ± probes. Do not replace Phase 1 Gaussian mutations with structural sweeps.
