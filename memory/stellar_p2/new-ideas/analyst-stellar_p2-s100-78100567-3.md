# Analyst notes — stellar_p2-s100-78100567 @ 38 candidates
## What the search is doing
The search has been trapped in a regression loop for over 15 candidates (c0022–c0038). The writers are repeatedly attempting to test orthogonal escape mechanisms (NAE generation, mode shifting, shallow ladders), but due to implementation fragility (eval budget starvation or dictionary formatting bugs), the evaluation logic fails and falls back to a hardcoded base matrix. The base matrix itself lacks the two-stage depth contraction required to maximize $L$, so the returned boundary scores a static train ~0.6128. This is a pure structural failure of the evaluation loop.

## Binding problem(s) now
1. **Broken Eval Pipeline:** The candidates are evaluating zero NAE or structural variants successfully. They silently crash or exhaust the budget and fall back to the uncontracted base boundary.
2. **Feasibility-margin camping:** The previous campaign's high score (0.6400) was achieved strictly by pushing the aspect ratio to the tolerance wall. Buying margin back to improve the "honest score" has been proven physically destructive (`shallow-uncontraction-pitfall.md`).
3. **Missing basin diversity:** Every successful historical result is a $<1\%$ perturbation of a public seed (`provenance-and-independence.md`). Independent basins (like `nfp=2`) have been proposed repeatedly by analysts but never executed correctly.

## Decision: pivot — and why
**Continue** is dead because the local nfp=3 search space is completely exhausted and recent candidates cannot even evaluate correctly. **Revive** is useless because all abandoned solutions are just failed orthogonal perturbations on nfp=3.
I must **PIVOT** by cleanly executing the **nfp=2 NAE basin probe**. The physics objective scales as $L \propto A/\text{nfp}$. Reducing `nfp` from 3 to 2 theoretically provides a ~1.5x multiplier on $L$, presenting a massive potential category jump. By defining a robust, universally-typed boundary constructor (`_mk_bound`), we guarantee `eval_many` will not crash on dictionary key mismatches. If `nfp=2` fails to converge or violates QI constraints, the code safely falls back to the exact proven two-stage nfp=3 incumbent as the non-regressing floor.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Generate a diverse batch of nfp=2 `fm.seed_nae` boundaries, normalize their matrix dimensions dynamically, batch them alongside the proven nfp=3 incumbent, and select the best honest score.
**Mechanism:** 
1. Procedurally construct the guaranteed nfp=3 fallback using the exact B3 matrix with the proven two-stage contraction.
2. Generate 10 distinct nfp=2 NAE seeds targeting aspect ratios 8.0–9.5.
3. Use a dynamic matrix padder (`_pad`) to zero-pad all boundaries to identical $(8 \times 15)$ dimensions, preventing shape-ragged crashes in `eval_many`.
4. Safely append all candidates to a single `eval_many` list, explicitly enforcing the required schema.
5. Select using `honest_score` (prioritizing low feasibility margin without sacrificing physics).
**Expected effect:** If VMEC converges and QI holds for `nfp=2`, we will discover a completely novel basin with $L \gg 12.5$, leaving the 0.6361 leaderboard bar in the dust. If it fails, it safely locks in the 0.6128+ incumbent.

## Decision log (alternatives considered and rejected, with reasons)
- **Continue local nfp=3 perturbations (c0022+ style):** Rejected. The eval loop is structurally broken, and local perturbation axes (per-stage splits, m-localized bumps) are wiki-refuted.
- **Iterative SPSA / Nevergrad ascent:** Rejected. `spsa-ascent.md` proves iterative loops structurally betray the 240s time budget.
- **Cross-basin mode grafting:** Rejected. `mode-grafting-and-blends.md` shows splicing modes across `nfp` basins destroys VMEC spectral condensation.
