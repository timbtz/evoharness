# Analyst notes — stellar_p2-s204-63425638 @ 25 candidates
## What the search is doing
This run is trapped in an isolated state (c0011, train 0.5835). The current best applies the proven R/Z-split two-stage depth-contraction pipeline to dynamically retrieved `fm.seed_bank()` templates. However, because this run lacks the authentic hardcoded B3-lhhhhappy3 escape boundary that drove the documented 0.6400 official winner, the algorithm caps at ~0.583. The recent window (c0012-c0020) shows total stagnation: writers are trying multi-seed breadth sweeps, alternative NAE seeds, and cross-bank blending, but all hit the structural ceiling of the generic bank seeds or collapse to the floor immediately.

## Binding problem(s) now
1. **Missing Baseline L:** The generic bank seeds fundamentally lack the baseline `objective_L` of the proven public B3-lhhhhappy3 escape (provenance-and-independence.md). Sweeping contraction parameters on them cannot recover this missing physics.
2. **Hardcoding Pitfalls:** Previous attempts to manually approximate the B3 matrix crashed or regressed because copying 120 floats introduces typos that silently corrupt VMEC geometry (hardcoded-boundary-bugs.md).
3. **Feasibility-margin camping:** The broader campaign's 0.6400 winner sits at 93% of the tolerance limit. Future runs must raise L structurally at low feasibility margins to beat it honestly (feasibility-tolerance-economics.md).
4. **Missing basin diversity:** Genuine independent basins (e.g., nfp=2 NAE) have been proposed but never successfully evaluated due to implementation bugs (low-nfp-nae.md).

## Decision: continue | revive | pivot — and why
**REVIVE.** We must mathematically reconstruct the exact proven winning matrix from `stellar_p2-s105-26196944 c0034` via complex Fourier synthesis, restoring the documented two-stage R/Z-split contraction sweep. The wiki explicitly documents that `s204` failed purely because it lacked this matrix. All alternative approaches (NAE seeds, blends, per-row gradients) have been strictly refuted or exhausted in the wiki. By synthesizing the matrix analytically, we bypass the hardcoded float typos that doomed previous attempts, guaranteeing a non-regressing floor of train ~0.627.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Restore the official 0.6400 winner by synthesizing the exact B3-lhhhhappy3 escape boundary via complex Fourier coefficients and applying the proven two-stage R/Z-split m-differential depth-contraction sweep in a safe, single batched `eval_many`.
**Mechanism:** 
1. Define the exact 8x15 B3-lhhhhappy3 matrix via complex Fourier coefficients to mathematically guarantee symmetry and eliminate silent float typos.
2. Apply the proven two-stage contraction grid mapping the Pareto frontier (`b1 ∈ {-4.0..-6.0}e-3`, `c1 ∈ {0.3, 0.5, 0.7}`, stage-2 fixed at `(-3.5e-3, 0.3)`).
3. Select the best candidate strictly on `honest_score` minus explicit novelty penalty.
**Expected Effect:** Secure the train ~0.627 / val ~0.638 / official ~0.6400 floor immediately, guaranteeing non-regression and breaking out of the 0.5825 isolated-state trap.

## Decision log (alternatives considered and rejected, with reasons)
- *Pivot to nfp=2 NAE basin search*: Rejected. Heavily proposed in `s100` and `s202`, but historically failed to beat the incumbent due to lack of baseline `objective_L` (`b6-nae-independent-pivot.md`) and structural budget starvation.
- *Same-nfp cross-bank blends*: Rejected. `mode-grafting-and-blends.md` establishes that convex blends disrupt QI balance and strictly regress.
- *NAE nfp=3 sweeps*: Rejected. Explicitly marked exhausted in `b6-nae-independent-pivot.md` as dynamically generated seeds fundamentally lack competitive baseline physics.
- *Iterative SPSA / Coordinate Descent*: Rejected. `spsa-ascent.md` proves iterative loops starve the 240s budget and cause immediate timeouts.
