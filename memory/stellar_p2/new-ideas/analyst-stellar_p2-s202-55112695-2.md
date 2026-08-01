# Analyst notes — stellar_p2-s202-55112695 @ 25 candidates
## What the search is doing
The run is trapped at the local Pareto frontier of the nfp=3 B3-lhhhhappy3 basin, entirely relying on a hardcoded matrix that structurally caps train scores at ~0.612-0.622 instead of the 0.6269 floor due to float typos. Recent candidates confirm the search space for the two-stage R/Z split (`cr, cz = 0.5, 0.7`) is definitively saturated (`b3-contraction-pareto-sweep.md`). Micro-sweeps of stage parameters, localized m=2 perturbations, and orthogonal spectral tricks yield zero structural gains.

## Binding problem(s) now
1. **Feasibility-margin camping:** The entire historical margin over the leaderboard bar (0.6400 vs 0.6361) comes from spending the 1% aspect ratio tolerance, not from better physics (`feasibility-tolerance-economics.md`). The contraction ladder is exhausted.
2. **VLF blindness & missing basin diversity:** Every successful historical result is a $\le 0.5\%$ perturbation of the public #1 submission (`provenance-and-independence.md`). The campaign has never successfully evaluated an independent basin.

## Decision: continue | revive | pivot — and why
**PIVOT.** Continuing the contraction search on the nfp=3 basin is mathematically futile. The only open multiplicative lever for the target objective is field-period count: $L \propto A/N_{fp}$. With $A$ pinned at the ~10.10 aspect wall, reducing $N_{fp}$ from 3 to 2 theoretically provides a ~1.5x multiplier on $L$.

To guarantee the safety of this probe, I am bypassing the typo-prone hardcoded matrices entirely by using `fm.seed_bank(6)` as the non-regressing floor. To ensure the newly generated nfp=2 `fm.seed_nae()` candidates can be safely evaluated without key mismatches, I am dynamically enforcing strict VMEC symmetry (`r_cos[0, n<0] = 0`, `z_sin[0, n<=0] = 0`) and padding them uniformly to a safe `(8, 15)` shape before batching.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** Generate a batch of `nfp=2` NAE seeds, robustly normalize their matrix dimensions dynamically, and batch them alongside a guaranteed `seed_bank(6)` contraction floor.
**Mechanism:** If VMEC converges and QI holds for `nfp=2` at aspect $\sim 9.0$, the theoretical scaling law predicts $L \gg 12.5$, providing a category jump in score in a completely novel, low-feasibility basin. If QI fails, the `seed_bank` fallback safely lands at $\sim 0.620$ train.

## Decision log (alternatives considered and rejected, with reasons)
1. **REVIVE exact historical nfp=3 incumbent (`b1-rebase-and-escapes.md`)**: Rejected. The hardcoded matrices are severely vulnerable to silent float typos, capping the run at 0.613 instead of the 0.6269 floor. Using `seed_bank(6)` is explicitly proven to be a safer VMEC-valid fallback (`bank6-contraction-portfolio.md`).
2. **REVIVE nfp=3 NAE sweep (`b6-nae-independent-pivot.md`)**: Rejected. Explicitly marked exhausted because dynamically generated nfp=3 NAE seeds lack the baseline `objective_L` to clear the QI wall.
3. **CONTINUE micro-sweeping stage-1/2 grid parameters (`b3-contraction-pareto-sweep.md`)**: Rejected. Exhausted locally; the entire grid is strictly saturated with ~0.0007 of tolerance left.
