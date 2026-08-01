# Analyst notes — stellar_p2-s203-38950787 @ 12 candidates

## What the search is doing
The run is trapped in a typo-regression sinkhole around the hardcoded B3 escape matrix, with writers attempting micro-sweeps of the proven two-stage contraction grid. Every candidate (c0001 through c0010) relies on the same hardcoded `_B3_RCOS`/`_B3_ZSIN` matrices in the source text, but the run's absolute best train score is **0.6229** (c0000). This is a full **0.004 below** the documented expected train floor of 0.6269 (`c0045` from `s105-26196944`). The wiki explicitly warns (`ineffective-approaches/m2-localized-and-nae-pivots.md`) that this hardcoded payload is "extremely vulnerable to silent copy-paste typos during programmatic diffing," capping the run ~0.010 below the proven official score. The writers have spent 12 candidates fruitlessly tweaking stage parameters without realizing their base boundary is structurally corrupted relative to the real champion.

## Binding problem(s) now
1. **Missing basin diversity & Feasibility-margin camping:** The entire run is camping at ~93% of the aspect ratio tolerance wall (feasibility ~0.009+), blindly micro-polishing a cosine 0.999989 near-copy of public seed #0 (`provenance-and-independence.md`). The official scoring system heavily penalizes this with the novelty penalty, and normalized to equal tolerance, this boundary is actually ~0.004 below the leaderboard leader.
2. **Eval starvation & VLF blindness:** The 12-27s/eval cost on hardcoded authentic high-resolution boundaries annihilates the budget. Large grids time out, leaving no budget to verify candidates at low fidelity, meaning the optimizer is entirely blind to train-identical perturbations.

## Decision: continue | revive | pivot — and why
**PIVOT.** Continuing the contraction search on the corrupted hardcoded B3 basin is mathematically futile (the local grid is fully mapped anyway, per `b3-contraction-pareto-sweep.md`). Furthermore, `low-nfp-nae.md` remains the only mathematically grounded lever for a category jump, explicitly noting that the nfp=2 NAE basin has **never been successfully evaluated** due to repeated historical code failures (dead loops, formatting crashes).

We will abandon the hardcoded matrix entirely to eliminate typo regression and solve the eval starvation via a **dynamically generated multi-seed portfolio**. By pulling native VMEC-safe baselines from `fm.seed_bank(i)`, we guarantee cheap 1.5s evals. By including dynamically generated `fm.seed_nae(n_field_periods=2)` seeds, we safely and correctly probe the untested nfp=2 basin for the first time.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** A `seed_bank` + `seed_nae` multi-basin portfolio with strict dynamic zero-padding.
**Mechanism:**
1. Base candidates entirely on dynamically retrieved `fm.seed_bank(i)` and `fm.seed_nae(...)` to guarantee structurally clean, spectrally condensed VMEC baselines (eliminates hardcoded typo regressions).
2. Apply the proven two-stage R/Z contraction to multiple nfp=3 bank seeds.
3. Include uncontracted, lightweight nfp=2 `fm.seed_nae()` seeds to probe the theoretical $L \propto A/N_{fp}$ advantage.
4. **Crucial padding logic:** Functionally truncate/pad all matrices to a uniform safe resolution (e.g., 6x9) using native Python lists to avoid SciPy `ValueError` shape raggedness, and strictly enforce stellarator symmetry (`r_cos[0, n<0] = 0`, `z_sin[0, n<=0] = 0`).
5. Batch-evaluate all candidates in one `eval_many` call and sort by the novelty-penalized honest score.
**Expected effect:** Safely discovers a submittable boundary outside the 1e-3 export ball without starving the budget. If the nfp=2 NAE seed maintains QI feasibility, objective $L$ jumps to ~16-19 (score ≫ 0.64) in a novel basin. If it fails, it safely falls back to a dynamically-contracted bank seed.

## Decision log (alternatives considered and rejected, with reasons)
- **CONTINUE the local B3 contraction:** Rejected. The local search space is strictly mapped (`b3-contraction-pareto-sweep.md`), and the hardcoded matrix in the run is suffering from typo regressions that structurally cap the score at 0.6229.
- **REViVE `s105` c0045 matrix verbatim:** Rejected. Hardcoding 120 float literals guarantees future copy-paste typos in writer candidates, and high-resolution (mpol=7/8) boundaries cause massive VMEC timeouts and budget starvation.
- **SPSA / Iterative Ascent:** Rejected. Iterative loops structurally betray the batched `eval_many` design and consistently cause timeouts (`ineffective-approaches/spsa-ascent.md`).
