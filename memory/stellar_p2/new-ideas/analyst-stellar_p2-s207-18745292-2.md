# Analyst notes — stellar_p2-s207-18745292 @ 24 candidates
## What the search is doing
This run is in a strictly isolated state (bank seeds unavailable), attempting to build an independent QI basin from scratch. It is flailing in deep infeasibility. The incumbent (c0001) and recent attempts (c0020) sit at a catastrophic train score of -0.56 to -0.99 (maximum constraint violation ~0.5 to 1.0). The base `fm.seed_nae()` and `fm.seed_ellipse()` seeds start at enormous baseline violations, and the sequential SPSA loops lack the directional precision and budget to cross the 0.01 feasibility threshold.

## Binding problem(s) now
1. **Missing Baseline L / Infeasible Starts**: Dynamically generated NAE/ellipse seeds are completely uncompetitive. Proven per the wiki (`independent-basin-violation-descent.md`).
2. **Eval Starvation**: Iterative SPSA loops starve the 72-eval budget, while the raw search space is flat/degenerate relative to the random directions sampled.
3. **Feasibility-Margin Camping**: Even if the nfp=3 bank lineage were available, squeezing the aspect ratio is mathematically exhausted (`feasibility-tolerance-economics.md`).

## Decision: continue | revive | pivot — and why
**PIVOT.** We must implement a structured physics-first coordinate-descent directly on the Fourier coefficients. The random Gaussian perturbations in the SPSA loop are mathematically incapable of isolating the aspect ratio constraint. By strictly defining a coordinate basis that selectively scales $R_{1,0}$ (the major aspect-ratio-defining mode), we deterministically push the geometry toward the feasibility wall. This replaces eval-starved randomness with an analytically guided descent.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Physics-Guided Aspect-Ratio Descent via Structured Fourier Coordinate Polling.
**Mechanism:** Triage ~12 diverse NAE/ellipse seeds. Select the lowest-violation candidate. Instead of stochastic SPSA, execute a batched Coordinate Descent sweep that probes targeted structural moves: a dominant aspect-ratio reduction (scaling $R_{1,0}$), a low-mode spectral relaxation, and a major-radius shift. If any move decreases the violation, accept the best and continue the sequence deterministically.
**Expected Effect:** By directly targeting the dominant constraint instead of relying on random noise, this will break the -0.56 floor and yield a train score > -0.30, maximizing the utilization of the 72-eval budget.

## Decision log (alternatives considered and rejected, with reasons)
1. **CONTINUE (Random SPSA)**: Rejected. C0001-c0020 empirically prove random perturbations fail to locate the feasibility gradient.
2. **REVIVE (Contracted Bank Portfolio)**: Rejected. The run is explicitly isolated without `fm.seed_bank` access, per prior wiki notes on s204/s205/s206.
