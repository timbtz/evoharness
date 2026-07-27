# Memory index — stellar_p2 (ConStellaration P2, simple-to-build QI stellarator)

Claim ladder: beat ALM baseline 0.431 -> beat ExLLM 0.505 -> leaderboard #1 0.6361 -> raw official **0.6400**.
NOVELTY BAR: submittable = official > 0.6361 AND max-coeff distance >= 1e-3 from EVERY bank seed.
`fm.score()` is BLIND to the novelty penalty. Acceptance keys MUST explicitly subtract it.

**READ BEFORE CLAIMING THE WIN (2026-07-27 audit, both pages are load-bearing):**
- performance-analysis/feasibility-tolerance-economics.md — the 0.6400 sits at 93% of the official 1% feasibility tolerance; davidkh's 0.6361 sits at 7.5%. Exchange rate ~0.92 score per unit feasibility (three independent estimates agree). At equal margin we are ~0.632, i.e. **~0.004 BELOW the leader**, not above.
- performance-analysis/provenance-and-independence.md — every scoring result descends from a public submission. Pre-seed-bank runs (s42/s7/s11) ended 42-52x over tolerance at official **0.0**. The winner is davidkh's boundary at cosine 0.999989 (‖Δ‖/‖seed‖ 0.47%). B6-nae-independent has never been run.

Second, honesty-preserving leaderboard to track from now on: **best official score at feasibility ≤ 0.002**. Current holder: s17 c0005f, 0.6335 at feas 0.00144 (and it is a davidkh near-copy). Nothing we have beats 0.6361 on that board.

## Current Best
**`stellar_p2-s105-26196944` c0045: official/private 0.6400, val 0.6380, train 0.6269, feasibility 0.00931, bank_dist 2.64e-3.**
Independently re-verified 2026-07-27 from the archive in a clean container (0.6400 / 0.0093) and exported to `experiments/submissions/p2-3-dac057eed0c1b2a8-0.6400.json`; archive key `3-dac057eed0c1b2a8`. Runner-up c0088r1 (`3-024648d4008d3302`) re-verified at 0.6398 / 0.0096.
The program takes a hardcoded nfp=3 escape boundary (labelled "B3-lhhhhappy3" in-run, but by distance it is nearest davidkh) and deepens it with a TWO-STAGE composed R/Z-split m-differential contraction `[1+b1·(1+c1·(m-1))] · [1+b2·(1+c2·(m-1))]`. Jointly sweeping stage-1 depth (`b1`) and curvature (`c1`) walks the L/aspect-ratio Pareto frontier — which is also what spends the tolerance.

## successful-patterns
- b1-rebase-and-escapes.md — THE WIN: rebase off the pinned high-nfp seed to an nfp=3 escape (B1/B3, ~0.633) + R/Z-split depth contraction → official 0.6398 submittable.
- b3-contraction-pareto-sweep.md — Exploring the local Pareto frontier, two-stage depth composition, and joint stage-1/stage-2 parameter sweeps in a single batched `eval_many` safely locks in the best score.
- bank6-contraction-portfolio.md — Bank seed contraction sweep yields a stable VMEC-safe fallback score when hardcoded boundaries fail.

## ineffective-approaches
- alternative-m-profiles.md — Quadratic-in-m, quantized, piecewise, 3rd-stage concave, and power-law profiles disrupt QI geometry and regress.
- gaussian-spectral-noise.md — Random Gaussian noise disrupts QI balance without unlocking valid novelty.
- hardcoded-incorrect-escape.md — Hardcoding an inferior B3 escape matrix caps the score below the winner.
- spsa-ascent.md — Iterative nevergrad/SPSA/coordinate-descent ascent starves the eval budget and times out.
- m1-selective-contraction.md — m=1-row-selective differential scaling disrupts geometry and fails to beat the incumbent.
- top-bank-anisotropic-escape.md — `exp(-a*m - b*|n|)` scaling on the top bank seed collapses to the floor.
- mode-grafting-and-blends.md — Cross-basin/cross-nfp recombination and same-nfp homotopy regress.
- gradient-and-rotation-escapes.md — Random-subspace gradients, phase rotations, additive deltas, local ascent regress to floor.
- b4-deep-contraction.md — Standalone deep sweeps on B4 risk timeouts and fail to beat the tri-basin portfolio.
- b4-truncation-and-bank-escapes.md — Structural truncation, homotopy, and envelope escapes on B4/bank seeds collapse or time out.
- surrogate-and-nae-escapes.md — Batched quadratic surrogate steps and NAE-from-scratch basins fall back to the floor.
- toroidal-axis-contraction.md — Orthogonal n-axis, combined (m,n), m-gated n-axis, or `exp(-q|n|)` toroidal scaling disrupts geometry.
- spectral-and-depth-perturbations.md — Mid-m bumps, asymmetric base splits, and custom LF-gated loops plateau or timeout.
- b1-power-law-contraction.md — Nonlinear power-law profiles, additive high-m bands, and fabricated matrices plateau or crash.
- risoliao6-basin-extension.md — Extending the R/Z contraction sweep to structurally distinct bank seeds plateaus immediately.
- structural-axis-perturbations.md — Radial translation, independent NAE basins, and iterative coordinate descent fail or starve budget.
- constant-depth-b1-b2-splits.md — Sweeping b1/b2 splits at a fixed total depth sum fails to outperform the default split. (Deprecated: Merged into per-row-rz-and-3d-joint-sweeps.md)
- per-row-rz-and-3d-joint-sweeps.md — Per-row R/Z gradients, angular twists, stage-split cr variations, and multi-round joint 3D grids regress or time out.
- r0-rescale.md — Uniform major-radius (R0) rescale pre-contraction regresses by breaking aspect/QI coupling.
- m-transfer.md — Post-composition low-m to high-m zero-sum curvature transfer yields no gain over the optimal two-stage composition.

## performance-analysis
- feasibility-tolerance-economics.md — the official rule (`_DEFAULT_RELATIVE_TOLERANCE = 0.01`, max of 5 normalized violations, aspect ratio always binding) and the ~0.92 score-per-feasibility exchange rate: our margin over the bar is bought tolerance, not better physics.
- provenance-and-independence.md — pre-seed-bank runs never reached feasibility (official 0.0); every scoring result is a ≤0.5% perturbation of a public submission; B6-nae-independent never ran.

## implementation-insights
- budget-discipline.md — Slow evals cause timeouts; cap portfolios ≤15 evals in one batched eval_many.
- anchor-floor-rejection-bug.md — NEVER hardcode `base_b`/raw parent/diagonal as fallback; use the exact off-diagonal incumbent.
- lf-selection-bug.md — LF landmine gates must not override the train-score winner; use LF only as a tie-break.
- hardcoded-boundary-bugs.md — Never fabricate, manually truncate, or typo-corrupt Fourier matrices; VMEC strictly requires spectrally condensed shapes.

## new-ideas
- low-nfp-nae.md — Pivoting to an nfp=2 NAE seed to exploit the L ∝ A/Nfp scaling law for a category jump in score.
- untested-ascent-mechanisms.md — Absorbed analyst rounds (SPSA, DFO); mostly refuted/tried.
- analyst-stellar_p2-s105-26196944-2.md — Decision log @ 24 cands (predicted the c0022 breakthrough).
- analyst-stellar_p2-s105-72881323-*.md — Historical decision logs (3, 4, 5, 6, 8).
- analyst-stellar_p2-s105-26196944-3.md — in-run Opus analysis + decision log @ 36 cands (stellar_p2-s105-26196944).
- analyst-stellar_p2-s105-26196944-4.md — in-run Opus analysis + decision log @ 48 cands (stellar_p2-s105-26196944).

- analyst-stellar_p2-s105-26196944-5.md — in-run analysis + decision log @ 60 cands (stellar_p2-s105-26196944).
- analyst-stellar_p2-s105-72881323-3/4/5/6/8.md — historical decision logs (restored by index guard after a review dropped them).

## Open directions
1. **The contraction ladder is done as a scoring strategy.** It has ~0.0007 of tolerance left (0.00931 of 0.01) — at 0.92 score/feas that is ~0.0006 of score, and it buys nothing on the honest board. Stop re-parameterizing the local sweep.
2. **The only directions that still matter raise L at LOW feasibility**: a genuinely different basin (new-ideas/low-nfp-nae.md, nfp=2 NAE, L ∝ A/Nfp) or a global structural transform. Target: official > 0.6361 at feas ≤ 0.002.
3. **Run B6-nae-independent.** It is the only branch that tests whether this harness can reach a feasible QI boundary without a public seed — the open research question, and worth more than another 0.001 on the raw board.
4. TRUST val→official within a lineage: private 0.6400 > val 0.6380 > train 0.6269 (positive gap held on every winner).
