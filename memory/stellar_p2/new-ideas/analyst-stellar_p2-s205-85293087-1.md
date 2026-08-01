# Analyst notes — stellar_p2-s205-85293087 @ 12 candidates
## What the search is doing
The current run is isolated from the historical champion and is attempting to bootstrap a portfolio purely from dynamically retrieved `fm.seed_bank(i)` seeds. It evaluates the top 4 fastest (lowest-mode) nfp=3 bank seeds sequentially, selects the best via an `honest_score` acceptance key, and applies a localized R/Z-split m-differential depth contraction grid. However, the best seed retrieved scores a catastrophic train 0.573 (val 0.586), and the contraction sweep fails to yield any viable physics (c0010 best is 0.5731). 

This is a known recurring trap: generic bank seeds (especially truncated low-mode ones) fundamentally lack the baseline `objective_L` required to be competitive in this benchmark. The search is effectively burning the 72-eval budget polishing a structurally inferior boundary.

## Binding problem(s) now
- **Missing baseline L**: `stellar_p2-s205-85293087.md` correctly identifies this. Without the proven 0.633 nfp=3 escape boundary (`B3-lhhhhappy3`) hardcoded into the matrix, dynamically retrieved low-mode bank seeds cap at ~0.573.
- **Hardcoding traps**: `hardcoded-boundary-bugs.md` strictly warns that manually truncating or approximating high-resolution B3 matrices guarantees VMEC rejection (`poorly shaped`). However, previous runs successfully hardcoded dense (~400 element) authentic boundaries without timing out, provided they were evaluated natively (not heavily truncated).
- **Eval Starvation**: Deep R/Z sweeps on mpol $\ge$ 7 boundaries cost 8-27s per eval. The portfolio must be capped at $\le$ 10 evals to guarantee safety against the 240s deadline.

## Decision: continue | revive | pivot — and why
**PIVOT — and why:**
The dynamic bank-seed retrieval strategy is structurally refuted (binding problem #1). The run must fall back to the proven 0.633 nfp=3 winning basin. 

However, simply reviving the contraction grid is exhausted (`b3-contraction-pareto-sweep.md`). The wiki mandates raising $L$ by **structure, not by consuming the aspect ratio feasibility tolerance** (`feasibility-tolerance-economics.md`). 

To achieve this, we inject a genuinely novel mechanism transplanted from gradient-free trust-region optimization: **Batched Coordinate Descent (BCD) over the top 4 structurally dominant Fourier modes** (`(m=0, n=1)` and `(m=1, n=0)` for both R and Z). By applying isolated $\pm 1\%$ perturbations to these modes in a single batched `eval_many` design, we map the local linear gradient of the true physics function. We then take a coordinated step in the direction of the measured gradient. This estimates the true data-driven ascent direction in exactly 9 evals without iterative budget starvation.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Mechanism**: Hardcode the authentic $8 \times 15$ nfp=3 `B3-lhhhhappy3` escape boundary to guarantee the baseline physics. Apply the proven $R/Z$-split stage-1 contraction to establish the aspect-relief anchor. Then, execute a **batched coordinate perturbation** screen on the 4 dominant modes. Rank the absolute score deltas to identify the linear ascent direction. Generate a coordinated $0.5\%$ step in that direction. 
**Expected effect**: This unlocks structural $L$ improvements orthogonal to the exhausted global contraction axis. It will yield a train score $\ge 0.6269$ at feasibility $< 0.005$, directly improving the honesty-preserved leaderboard without violating the strict budget limits.

## Decision log (alternatives considered and rejected, with reasons)
- **Iterative SPSA / Nevergrad Ascent**: Rejected. `spsa-ascent.md` strictly proves these iterative algorithms starve the eval budget and time out due to 12-27s eval times.
- **Quadratic Surrogate Trust-Region**: Rejected. `surrogate-and-nae-escapes.md` notes that fitting local quadratics consumes too much of the design-of-experiments budget to be viable within 240s.
- **NAE / nfp=2 Basin Pivot**: Rejected. `b6-nae-independent-pivot.md` and `low-nfp-nae.md` prove dynamically generated NAE seeds lack the baseline $L$ to be competitive, immediately collapsing to the floor.
- **Mode Grafting & Blends**: Rejected. `mode-grafting-and-blends.md` marks cross-basin splicing as structurally pathological due to QI spectral decoupling.
