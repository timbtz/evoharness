# Analyst notes — stellar_p2-s100-78100567 @ 76 candidates
## What the search is doing
This run has spent 76 candidates almost entirely on variations of the proven R/Z-split two-stage contraction applied to the hardcoded nfp=3 B3 escape boundary. The current best in-run is c0057 (train 0.6135), but the run's train floor sits at ~0.6128 because of recurring float-typo corruption in the hardcoded matrix during programmatic diffing (`batch-key-and-silent-typos.md`). Recent candidates (c0052-c0060) have cycled between NAE seed probes, spectral reweighting, toroidal-mode damping, and incumbent re-evaluations — all returning to the ~0.6128 floor without improvement.

The campaign's best overall (`stellar_p2-s105-26196944` c0045) sits at official 0.6400 (feasibility 0.0093, 93% of the tolerance budget). At equal margin to the leaderboard leader davidkh (0.6361 at feas 0.0008), our winner is ~0.632 — **below** davidkh. The entire margin above 0.6361 is tolerance camping, not structural quality.

## Binding problem(s) now
1. **Feasibility-margin camping**: The contraction mechanism has saturated the aspect-ratio wall; there is ~0.0007 of tolerance left (`b3-contraction-pareto-sweep.md`).
2. **vlf blindness**: Recent candidates at train fidelity score identically at 0.6128, meaning changes are untested — the optimizer cannot distinguish them.
3. **Missing basin diversity**: Every scoring result in the campaign descends from a public seed-bank submission (provenance-and-independence.md). There is zero evidence the harness can find a feasible QI boundary independently. `B6-nae-independent` has never been run.
4. **NAE seeds lack baseline L**: Multiple attempts to pivot to nfp=2 or nfp=3 NAE seeds failed because they lack the baseline objective_L of the B3 escape.

## Decision: pivot — and why
The contraction axis on the B3 basin is provably exhausted (`b3-contraction-pareto-sweep.md` marks it "exhausted (locally)"). Every structural perturbation (per-row gradients, angular twists, mode grafting, spectral bumps, phase rotations, m-transfers, R0 rescales, uncontraction ladders) has been refuted. The only open mathematical lever is a genuinely different basin (`low-nfp-nae.md`).

The nfp=2 NAE pivot was proposed but never properly tested — prior attempts botched the batch format or had dead NAE loops (`dead-nae-loop-bug.md`, `stellar_p2-s100-78100567.md`). The theoretical basis is sound: L ∝ A/Nfp means at aspect ratio 10 with nfp=2 vs nfp=3, L should jump by ~50% if QI holds. The recent paper by Landreman & Jorge (https://arxiv.org/pdf/2509.16320) confirms that near-axis expansions can produce spectrally-condensed QI boundaries directly, and that single-step optimization on the full Fourier spectrum (ESS scaling) eliminates staging decisions.

The pivot: generate nfp=2 NAE seeds with aggressive QI targets, apply the proven two-stage R/Z contraction to push aspect toward the wall, and batch-evaluate alongside the incumbent floor. If QI holds at nfp=2 with L > 13, this is a category jump. If it fails, we return the incumbent floor safely.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**nfp=2 NAE basin probe with depth contraction.** Generate 8 nfp=2 NAE seeds via `fm.seed_nae()` with varying aspect ratios (8.0-10.0), rotational transforms (0.4-0.6), and mirror ratios (0.10-0.18). Pad to (mpol=8, ntor=7) to match the B3 matrix shape. Apply the proven two-stage R/Z contraction at 3 depth levels. Batch-evaluate all candidates + the incumbent floor in one `eval_many`. Select by honest_score + novelty penalty. If any nfp=2 seed achieves honest_score > 0.6128 (the incumbent), select it; otherwise return the incumbent.

Expected effect: Either (a) at least one nfp=2 seed lands feasible with L > 13 → score > 0.65 in a novel basin, or (b) nfp=2 seeds are infeasible/QI-broken and we return the incumbent floor. The single empirical question is whether QI residual stays below -4 at nfp=2.

## Decision log (alternatives considered and rejected, with reasons)
1. **CONTINUE contraction variations on B3**: REJECTED. The wiki marks this "exhausted (locally)" — all stage parameter corners, cr/cz splits, depth compositions, and curvature profiles are saturated (`b3-contraction-pareto-sweep.md`, `alternative-m-profiles.md`, `per-row-rz-and-3d-joint-sweeps.md`).
2. **REVIVE c0057's sharper-tuned two-stage contraction**: REJECTED. c0057 (train 0.6135) barely improved over the floor and the mechanism is the same saturated axis.
3. **REVIVE mode-grafting between bank seeds**: REJECTED. Cross-basin and same-nfp blends are marked "exhausted" — they destroy QI coupling (`mode-grafting-and-blends.md`).
4. **PIVOT to surrogate-assisted DFO**: REJECTED. Quadratic surrogate steps were tried and failed (`surrogate-and-nae-escapes.md`). Budget (72 evals) is too small for iterative model-building.
5. **PIVOT to nfp=2 NAE (CHOSEN)**: The only mathematically grounded lever for a category jump. Prior implementation failures were bugs, not refutations of the idea itself. The nfp=2 basin has never been properly evaluated in a batched design with the contraction mechanism.
