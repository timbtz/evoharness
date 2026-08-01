# Analyst notes — stellar_p2-s207-18745292 @ 12 candidates
## What the search is doing
The run is blindly flailing in an uncompetitive basin. The incumbent (c0001) and recent attempts (c0002-c0010) rely on dynamically generated `fm.seed_nae()` and `fm.seed_ellipse()` baselines evaluated via sequential SPSA loops or structural grids. As documented extensively in the wiki (`b6-nae-independent-pivot.md`, `surrogate-and-nae-escapes.md`), dynamically generated NAE/ellipse seeds fundamentally lack the baseline `objective_L` required to clear the QI wall. As a result, the run is trapped at a catastrophic train score of -0.56 to -0.93. 

## Binding problem(s) now
1. **Missing Baseline L**: The dynamic seeds sit at a maximum constraint violation of ~0.5 to 1.0. No amount of SPSA or structural perturbation can bridge the physics gap to feasibility (~0.01) under the strict eval budget.
2. **Eval Starvation**: The sequential SPSA loops in c0001-c0004 starved the budget, costing ~8-12 seconds per eval while failing to find any feasible points.
3. **Feasibility-Margin Camping**: To beat davidkh's 0.6361 honorably, we must raise L by physical structure rather than squeezing the aspect ratio wall. The historical nfp=3 basin is exhausted in this regard.

## Decision: continue | revive | pivot — and why
**REVIVE.** The lowest-risk, highest-yield action is to dynamically retrieve the top nfp=3 public bank seeds using `fm.seed_bank()` and apply the proven multi-seed bank contraction portfolio. 

The wiki proves (`bank6-contraction-portfolio.md`) that retrieving top nfp=3 bank seeds provides a perfectly convergent VMEC-safe baseline. From there, applying the R/Z-split m-differential structural contraction guarantees a positive baseline score (~0.62 train) and safely provides structural novelty (`bank_dist > 1e-3`), breaking the -0.5 floor trap without risking hardcoded matrix transcription bugs.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Batched nfp=3 Bank Portfolio Sweep with R/Z-Split Structural Contraction.
**Mechanism:** Triage `fm.seed_bank_info()` for the highest-scoring `nfp=3` seeds. Batch a 15-candidate portfolio in a single `eval_many` call, applying an R/Z-split m-differential contraction sweep (`base ∈ [-2e-3, -6e-3]`, `curv ∈ [0.0, 0.5]`). Strictly enforce VMEC symmetry patterns using boolean masks derived directly from the native seed. Select the top candidate using a novelty-penalized acceptance key (`key = honest_score - 0.05 * max(0.0, 1.0 - fm.bank_dist(b) / 1e-3)`). 
**Expected Effect:** Breaks the -0.56 floor safely by adopting a competitive baseline, yielding a train score > 0.60 with strictly valid `bank_dist > 1e-3` while leaving the eval budget intact.

## Decision log (alternatives considered and rejected, with reasons)
1. **CONTINUE (SPSA / NAE grids)**: Rejected. c0001-c0010 empirically prove NAE seeds lack baseline physics. SPSA starves the budget and caps at -0.56.
2. **REVIVE (Exact hardcoded 120-float B3 matrix)**: Rejected. Hardcoding long float matrices introduces silent 15-digit transcription typos that silently corrupt the incumbent (`batch-key-and-silent-typos.md`).
3. **PIVOT (Boundary-Domain Contraction / nfp=2 transport)**: Rejected. nfp=2 NAE basin jumps lack baseline L and collapse instantly (`b6-nae-independent-pivot.md`). The immediate priority is escaping the sub-zero floor safely.
