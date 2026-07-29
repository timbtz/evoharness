# Analyst notes — stellar_p2-s100-78100567 @ 51 candidates
## What the search is doing
The search is deadlocked. The current best candidate in this run is c0016 (train 0.6229, val 0.6248). Every recent candidate (c0032r1 through c0040) has either crashed with `SyntaxError` or regressed to train ~0.6128 by failing to correctly append dynamically generated boundaries to the `eval_many` batch (see `dead-nae-loop-bug.md`). Writers repeatedly attempt to pivot to `nfp=2` but introduce structural bugs (`KeyError` on missing dict keys, malformed matrix shapes, silent fallbacks to uncontracted base matrices), completely wasting the eval budget and yielding the exact same hardcoded fallback. 

## Binding problem(s) now
1. **Broken Eval Pipeline:** Candidates evaluating `seed_nae` dictionaries fail to match the strict `eval_many` schema, or fail to pad matrices to identical shapes, causing `eval_many` to crash and fall back to a score-destroying boundary.
2. **Feasibility-margin camping:** The campaign's official best (0.6400) achieves its score entirely by pushing the aspect ratio to 93% of the tolerance wall. It is structurally blocked from further improvement.
3. **Missing basin diversity:** Every successful historical result is a $<0.5\%$ perturbation of a public seed (`provenance-and-independence.md`). The `nfp=2` NAE basin—which theoretically scales $L \propto A/\text{nfp}$ and could provide a category jump from 12.5 to ~18.5—remains **entirely untested** due to the recurring implementation bugs.

## Decision: continue | revive | pivot — and why
**PIVOT — and why:** 
- **Continue** is dead. The local `nfp=3` search space is fully exhausted, and c0049's m=2 localized perturbations merely confirmed the Pareto wall.
- **Revive** is useless. All abandoned solutions in this campaign are structurally identical perturbations of the `nfp=3` public seed. 
- **Pivot** is mandatory. The ConStellaration P2 paper (arXiv:2506.19583) establishes that the physics objective scales as $\tilde{L}_{\nabla B} \propto A/\text{nfp}$. Since aspect ratio ($A$) is pinned at the ~10.10 wall, reducing `nfp` from 3 to 2 theoretically provides a ~1.5x multiplier on $L$. The downside is exactly zero (we hardcode a guaranteed non-regressive floor), and the upside is a category jump in score, a low-feasibility physics structure, and a genuinely novel independent basin. Past attempts at this pivot failed purely due to code bugs (ragged arrays, missing keys).

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Generate a batch of `nfp=2` NAE seeds, robustly normalize their matrix dimensions dynamically to a fixed $(8 \times 15)$ grid, and batch them alongside the guaranteed `nfp=3` incumbent floor. 
**Mechanism:** 
1. Procedurally construct the guaranteed `nfp=3` fallback using the exact B3 matrix with the proven two-stage R/Z-split contraction.
2. Generate 10 distinct `nfp=2` NAE seeds targeting aspect ratios 8.0–9.5.
3. Use a strict matrix padder (`_pad`) to zero-pad all NAE boundaries to $(8 \times 15)$ dimensions and enforce exact dictionary schema keys.
4. Safely append all candidates to a single `eval_many` list, guaranteeing no `eval_many` crashes.
5. Select the best candidate by `honest_score` (prioritizing low feasibility margin without sacrificing physics).
**Expected effect:** If VMEC converges and QI holds ($\log_{10}(\text{qi}) \le -4$), an `nfp=2` seed will jump to $L \approx 16-18$ (score $\gg 0.64$) in a completely novel basin. If it fails, it safely locks in the 0.6229+ incumbent.

## Decision log (alternatives considered and rejected, with reasons)
- **Continue local nfp=3 perturbations (c0049 style):** Rejected. The eval loop is structurally broken, and local perturbation axes (per-stage splits, m-localized bumps) are wiki-refuted.
- **Iterative SPSA / Nevergrad ascent:** Rejected. `spsa-ascent.md` proves iterative loops structurally betray the 240s time budget.
- **Cross-basin mode grafting:** Rejected. `mode-grafting-and-blends.md` shows splicing modes across `nfp` basins destroys VMEC spectral condensation.
