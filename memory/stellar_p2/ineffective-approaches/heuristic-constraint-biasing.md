# Hardcoded Heuristic Constraint Biasing
Injecting hardcoded directional bias vectors or constraint-aware selections into mutations to simultaneously target constraints destroys the mathematically smooth search space and blindly violates physics geometry.

## How it was tried
- stellar_p2-s7-30186401 c0008 (-0.813): Introduced `_viols()` metrics reading and `_mutate()` bias vectors. Pushed `z` and `r` in anti-phase for mirror/elongation, and pumped `ntor >= 2` modes for QI.
- stellar_p2-s7-30186401 c0007 (-0.753): Expanded the NAE seed matrix and pushed constraint-aware biases aggressively in parallel streams.
- stellar_p2-s11-38566380 c0003 (-0.627): Added constraint-aware elite pool tracking per-constraint violations and rotating the "best" elite based on weighted (1-violation) roulettes.
- stellar_p2-s11-38566380 c0007 (-0.513): Replaced Phase 1 random Gaussian with hardcoded single-axis probes on m=1 poloidal coefficients to directly attack elongation/mirror-ratio.

## Why it failed
Stellarator boundaries must remain smooth NAE approximations. Hardcoding directional steps directly introduces tremendous geometric kinks, crushing the boundary's aspect ratio. Furthermore, constraint-aware selection logic often misreads raw metric values (e.g., aspect ratio ~8) as normalized violations, collapsing roulette selection to a uniform distribution that accomplishes nothing but perturbing the RNG call sequence. 

## Verdict
refuted — Never replace gradient descent / CMA-ES with hardcoded physics heuristics. If targeting specific constraints, use an objective proxy, not direct coordinate manipulation.
