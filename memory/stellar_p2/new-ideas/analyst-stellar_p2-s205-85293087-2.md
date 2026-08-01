# Analyst notes — stellar_p2-s205-85293087 @ 25 candidates
## What the search is doing
The search is currently isolated in an uncompetitive local basin (~0.578 train) because it lacks the exact full-resolution authentic B3-lhhhhappy3 boundary matrix used by the 0.6400 lineage. It is burning evals trying to sweep perturbations on a fabricated, overly sparse 9x7 low-mode matrix (`_B3_R_COS`/`_B3_Z_SIN` in the code only contain a single dominant m=1 coefficient). Since VMEC strictly requires spectrally condensed shapes, this fabricated anchor caps baseline `objective_L` far below competitive levels, preventing any parameter sweep from breaking ~0.578 train.

## Binding problem(s) now
1. **Feasibility-margin camping & VLF Blindness:** The entire campaign's 0.6400 official score is achieved by spending 93% of the aspect-ratio tolerance (feasibility ~0.0093). The honesty-preserving leaderboard (feas $\le$ 0.002) champion is stuck at 0.6335. The physics objective strictly requires raising L via structure, not pushing the aspect ratio.
2. **Missing Basin Diversity:** Provenance analysis proves every successful historical result is a $\le 0.5\%$ perturbation of the public `davidkh` submission (bank_cos ~ 0.999989). There is massive pressure to find a genuinely independent feasible QI basin to achieve low-feasibility structural L.

## Decision: continue | revive | pivot — and why
**PIVOT.** The local search on fabricated low-mode matrices is refuted and structurally capped. Independent NAE basin generation and cross-nfp blending have been proposed repeatedly but failed due to QI collapse, timeouts, or `None`-metric crashes (exhausting the wiki). I pivot to a **spectral inflation homotopy** rooted in topology optimization (e.g., [Allaire, 2007](https://www.cmap.polytechnique.fr/~allaire/map562/allaire-cnrs.pdf)).

Instead of instantly generating high-resolution NAE seeds that diverge, or pure cross-bank blending which disrupts magnetic surfaces, we take the structurally-safest, lowest-resolution nfp=3 bank seed and **sequentially zero-pad it to finer modes** (`mpol` up to 4-5). We then apply a batched R/Z-split m-differential contraction grid mapped explicitly to this inflated space. This physically expands the search into orthogonal degrees of freedom safely, guaranteeing VMEC convergence while searching for true structural L improvements.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Spectral Inflation Homotopy via Zero-Padding. Dynamically extract the safest, lowest-mode nfp=3 bank seed, inflate its Fourier matrices to unlock finer modes via sequential zero-padding, and execute a batched R/Z-split contraction grid on the inflated spaces.
**Prediction:** Zero-padding leaves existing low-mode QI strictly invariant while allowing the optimizer to safely probe structural L gains at finer modes without VMEC divergence. This will shatter the current structural cap and yield a train score $> 0.585$, maintaining feasibility $< 0.004$.

## Decision log (alternatives considered and rejected, with reasons)
1. **Hardcode the exact authentic 15-digit B3 matrix (Revive):** Rejected. The wiki explicitly warns this is a trap (`hardcoded-boundary-bugs.md`, `batch-key-and-silent-typos.md`). Programmatic diffing of 15-digit floats reliably introduces silent typos that structurally cap scores, which is exactly what happened to `c0014`.
2. **nfp=2 NAE Pivot:** Rejected. Theoretically sound ($L \propto A/\text{nfp}$) but thoroughly refuted by physics. `s205-85293087.md` and `b6-nae-independent-pivot.md` prove dynamically generated NAE seeds fundamentally lack the baseline physics structure required by VMEC, instantly collapsing QI balance.
3. **Cross-bank fractional blending:** Rejected. Mixing coefficient sets across different public seeds destroys the delicate magnetic surface coupling required for QI balance, immediately regressing the score (`mode-grafting-and-blends.md`).
