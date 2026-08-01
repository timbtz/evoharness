# Analyst notes — stellar_p2-s204-63425638 @ 12 candidates
## What the search is doing
This run is trapped in an isolated state without the authentic hardcoded nfp=3 boundary. The current best (c0001, train 0.5825, val 0.5946) attempts to execute the proven R/Z-split depth-contraction sweep dynamically on `fm.seed_bank()` templates. Because generic bank seeds lack the heavily optimized baseline `objective_L` of the hardcoded B3-lhhhhappy3 boundary, the algorithm caps at 0.5825 instead of the documented ~0.627 floor. Subsequent attempts (c0005-c0010) tried NAE pivots or multi-seed blends, but all either regressed, timed out, or fell back to this 0.5825 floor.

## Binding problem(s) now
1. **Feasibility-margin camping:** The incumbent 0.6400 official score sits at 93% of the tolerance limit. Future runs must raise L structurally at low feasibility to beat it honestly.
2. **VLF blindness:** Train fidelity is blind to micro-perturbations, meaning standard local search methods are untested without direct low-fidelity verification, starving the budget.
3. **Eval starvation:** Iterative or large-batch (>15) evaluations on high-resolution or poorly condensed boundaries consume the 240s CPU budget, leading to timeouts.
4. **Missing basin diversity:** Every high-scoring boundary derived from a public seed is a ≤0.5% perturbation. Genuine independent basins (e.g., nfp=2 NAE) have been proposed but never successfully evaluated due to implementation bugs.

## Decision: continue | revive | pivot — and why
**REVIVE.** We must inject the exact proven hardcoded matrix from `stellar_p2-s105-26196944 c0034` (the 0.6400 official winner) and restore the proven two-stage R/Z-split batched sweep. The wiki explicitly documents that `s204` failed purely because it lacked this matrix ("the algorithmic mechanism is proven, but its success is locked behind the specific hardcoded boundary"). All alternative approaches (NAE seeds, blends, per-row gradients) have been strictly refuted or exhausted in the wiki. 

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Restore the official 0.6400 winner by hardcoding the exact B3-lhhhhappy3 escape boundary and applying the proven two-stage R/Z-split m-differential depth-contraction sweep in a safe, single batched `eval_many`.
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
