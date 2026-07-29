# Analyst notes — stellar_p2-s100-78100567 @ 12 candidates
## What the search is doing
This run (`stellar_p2-s100-78100567`) was tasked with the open research goal: pivoting away from the exhausted nfp=3 contraction axis to find a genuinely novel basin, specifically probing **nfp=2 NAE seeds** to exploit the `L ∝ A/Nfp` scaling law. However, due to repeated implementation failures, the run is trapped in a loop of just evaluating slightly shallower variants of the hardcoded nfp=3 `B3-lhhhhappy3` escape boundary. 

The current best candidate in this run is c0002 (train 0.6136, val 0.6246). It attempts to find a low-feasibility (honest) boundary by applying shallower two-stage contractions. Every attempt to actually generate and evaluate the nfp=2 NAE seeds (c0005 through c0010) crashed (`SyntaxError`, dead code loops) or regressed to the floor because the NAE seeds were either omitted from the `eval_many` batch or malformed.

## Binding problem(s) now
1. **Dead NAE Loop & Evaluation Starvation:** The writer candidates have continuously failed to append NAE seeds to the `eval_many` list correctly, causing syntax errors or silent fallbacks. The 72-eval budget is being consumed by hardcoded nfp=3 matrices that evaluate slowly (~8-12s/eval), starving the search.
2. **Feasibility-margin Camping:** The search is still trying to squeeze fractions of a percent out of the aspect-ratio tolerance on the nfp=3 basin, which the wiki explicitly flags as exhausted (`feasibility-tolerance-economics.md`). The entire local perturbation space (m-profiles, n-axis, splits, twists) is refuted.
3. **Missing Basin Diversity:** We have zero successful evaluations of an nfp=2 boundary. The entire campaign rests on nfp=3/4 perturbations of public submissions.

## Decision: pivot — and why
**Continue is dead** (the nfp=3 contraction space is mathematically and empirically exhausted). **Revive is useless** (all abandoned solutions in this run are just failed NAE executions that fell back to the inferior nfp=3 floor). 

I must **PIVOT** by injecting the correct implementation of the nfp=2 NAE basin probe. The ConStellaration paper (arXiv:2506.19583) establishes that the physics objective scales as `L̃_∇B ∝ A/Nfp`. Since aspect (A) is pinned at the ~10.1 wall, reducing `Nfp` from 3 to 2 theoretically provides a ~1.5x multiplier on L. The downside is zero (we hardcode a safe nfp=3 fallback), and the upside is a category jump in score plus a genuinely novel basin.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Dynamically generate a tight batch of nfp=2 `fm.seed_nae` boundaries, append them correctly to a single batched `eval_many` call, and select the best. 
**Mechanism:** 
1. Procedurally construct the guaranteed nfp=3 fallback (the proven two-stage contraction on B3) to ensure a non-regressing baseline (~0.613).
2. Generate 8 distinct nfp=2 NAE seeds using `fm.seed_nae(n_field_periods=2, aspect_ratio=8.0..9.5, max_poloidal_mode=4..5)`.
3. Batch all candidates together in one `fm.eval_many` call to respect the budget.
4. Select using the `honest_score` combined with the novelty penalty (`fm.bank_dist`).
**Expected effect:** If VMEC converges and QI holds (`log10_qi <= -4`), an nfp=2 seed at aspect 9.0 will jump to L ≈ 16-18 (score ≫ 0.64) in a completely novel basin. If it fails, it safely locks in the 0.613 fallback without crashing.

## Decision log (alternatives considered and rejected, with reasons)
- **Continue shallow nfp=3 contraction (c0002 style):** Rejected. The wiki (`shallow-contraction-for-honest-margin.md`) proves this just trades score for feasibility without finding a better Pareto frontier.
- **Iterative SPSA/Nevergrad ascent:** Rejected. The wiki (`spsa-ascent.md`) confirms iterative loops annihilate the budget due to 8-27s eval times.
- **Cross-basin mode grafting:** Rejected. The wiki (`mode-grafting-and-blends.md`) shows splicing modes across nfp basins destroys the delicate spectral condensation required by VMEC.
