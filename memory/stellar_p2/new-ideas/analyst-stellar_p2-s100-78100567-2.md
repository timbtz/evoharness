# Analyst notes — stellar_p2-s100-78100567 @ 25 candidates
## What the search is doing
The search has run 25 candidates but is trapped in an endless local-perturbation loop. The incumbent relies on a hardcoded `B3-lhhhhappy3` nfp=3 escape boundary. Recent attempts (c0008-c0025) consist of probing fundamentally saturated local axes: primal-dual interpolation, additive R-offsets, localized m=2 perturbations (c0016), sheared toroidal scaling (c0018), and anisotropy splits (c0019). Every mechanism either tied the incumbent exactly or regressed. The search is strictly camping on the nfp=3 feasibility-tolerance Pareto frontier (raw 0.6400, but ~0.632 honest score at 93% feasibility spend), failing to discover any new physics or structural lever.

## Binding problem(s) now
1. **Feasibility-margin camping**: The "winning" 0.6400 score strictly descends from a public seed (davidkh). The entire margin over the leaderboard comes from spending aspect-ratio tolerance. The search is spinning its wheels micro-tuning it.
2. **VLF blindness & Eval starvation**: Slow VMEC evals (~8-12s for mpol=8 boundaries) cause massive budget starvation. The wiki explicitly restricts portfolios to $\le 15$ evals in a single `eval_many` batch, starving iterative search.
3. **Missing basin diversity**: Provenance analysis shows *every* scoring result is a $\le 0.5\%$ perturbation of a public seed. The harness has never evaluated an independent basin. Specifically, nfp=2 NAE seeds have been proposed multiple times but *always* failed in code generation (silent fallbacks, dead loops, or malformed batch dictionaries).

## Decision: pivot — and why
**Continue** is dead: the local nfp=3 search space is proven exhausted across 15+ rejected mechanisms. **Revive** is useless: all abandoned solutions in this run are just nfp=3 contractions that regress or tie.
I must **PIVOT** by injecting the correct implementation of the **nfp=2 NAE basin probe**. The ConStellaration P2 objective scales theoretically as $\tilde{L}_{\nabla B} \propto A/\text{nfp}$. With aspect ratio ($A$) pinned at the ~10.10 wall, reducing nfp from 3 to 2 provides a ~1.5x multiplier on $L$. The downside is exactly zero (we hardcode the nfp=3 incumbent as a fallback), and the upside is a category jump in score, low-feasibility physics structure, and a genuinely novel independent basin. Past attempts at this pivot (c0001, c0002) failed purely due to implementation bugs (malformed `eval_many` batches).

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Generate nfp=2 NAE seeds using `fm.seed_nae()` and batch them correctly alongside the guaranteed nfp=3 incumbent.
**Mechanism:** 
1. Procedurally construct the guaranteed nfp=3 fallback (the proven two-stage contraction on B3) to ensure a non-regressing baseline (~0.613 train floor).
2. Generate 10 distinct nfp=2 NAE seeds across aspect ratios 8.0–9.5 and different rotational transforms.
3. Safely append all candidates to a single `eval_many` list, ensuring dictionary keys exactly match the hardcoded schema (`r_cos`, `z_sin`, `r_sin`, `z_cos`, `n_field_periods`, `is_stellarator_symmetric`).
4. Select using the `honest_score` combined with the novelty penalty (`fm.bank_dist`).
**Expected effect:** If VMEC converges and QI holds ($\log_{10}(\text{qi}) \le -4$), an nfp=2 seed will jump to $L \approx 16-18$ (score $\gg 0.64$) in a completely novel, low-feasibility basin. If it fails, it safely locks in the 0.613 fallback without crashing or timing out.

## Decision log (alternatives considered and rejected, with reasons)
- **Continue shallow nfp=3 contraction (c0016 style):** Rejected. The wiki (`shallow-contraction-for-honest-margin.md`) proves this just trades score for feasibility without finding a better Pareto frontier.
- **Iterative SPSA/Nevergrad ascent:** Rejected. The wiki (`spsa-ascent.md`) confirms iterative loops structurally betray the budget due to 8-27s eval times.
- **Cross-basin mode grafting:** Rejected. The wiki (`mode-grafting-and-blends.md`) shows splicing modes across nfp basins destroys the delicate spectral condensation required by VMEC.
