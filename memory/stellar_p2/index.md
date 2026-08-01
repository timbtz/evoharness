# Memory index — stellar_p2 (ConStellaration P2, simple-to-build QI stellarator)

Claim ladder: beat ALM baseline 0.431 -> beat ExLLM 0.505 -> leaderboard #1 0.6361 -> raw official **0.6400**.
NOVELTY BAR: submittable = official > 0.6361 AND max-coeff distance >= 1e-3 from EVERY bank seed.
`fm.score()` is BLIND to the novelty penalty. Acceptance keys MUST explicitly subtract it.

**READ BEFORE CLAIMING THE WIN (2026-07-27 audit, both pages are load-bearing):**
- performance-analysis/feasibility-tolerance-economics.md — the 0.6400 sits at 93% of the official 1% feasibility tolerance; davidkh's 0.6361 sits at 7.5%. Exchange rate ~0.92 score per unit feasibility. At equal margin we are ~0.632.
- performance-analysis/provenance-and-independence.md — every scoring result descends from a public submission. B6-nae-independent has never been run.

Second, honesty-preserving leaderboard to track from now on: **best official score at feasibility ≤ 0.002**. Current holder: s17 c0005f, 0.6335 at feas 0.00144. Nothing we have beats 0.6361 on that board.

## Current Best
**`stellar_p2-s105-26196944` c0045: official/private 0.6400, val 0.6380, train 0.6269, feasibility 0.00931, bank_dist 2.64e-3.**

## successful-patterns
- structural-contraction.md — THE WIN: R/Z-split structural contraction safely expands the L/feasibility Pareto frontier.
- b1-rebase-and-escapes.md — Rebase off the pinned high-nfp seed to an nfp=3 escape (B1/B3, ~0.633) + R/Z-split depth contraction → official 0.6398 submittable.
- b3-contraction-pareto-sweep.md — Exploring the local Pareto frontier, two-stage depth composition, and joint stage-1/stage-2 parameter sweeps.
- bank6-contraction-portfolio.md — Bank seed contraction sweep yields a stable VMEC-safe fallback score.

## ineffective-approaches
- alternative-m-profiles.md — Quadratic-in-m, quantized, piecewise, 3rd-stage concave, and power-law profiles disrupt QI geometry and regress.
- gaussian-spectral-noise.md — Random Gaussian noise disrupts QI balance without unlocking valid novelty.
- hardcoded-incorrect-escape.md — Hardcoding an inferior B3 escape matrix caps the score below the winner.
- spsa-ascent.md — Iterative nevergrad/SPSA/coordinate-descent ascent starves the eval budget and times out.
- m1-selective-contraction.md — m=1-row-selective differential scaling disrupts geometry and fails to beat the incumbent.
- top-bank-anisotropic-escape.md — `exp(-a*m - b*|n|)` scaling on the top bank seed collapses to the floor.
- mode-grafting-and-blends.md — Cross-basin/cross-nfp recombination and same-nfp homotopy regress.
- isobaric-and-toroidal-modes.md — Fixed-m isobaric rotations and toroidal n-neighborhood selective bumps regress.
- b4-deep-contraction.md — Standalone deep sweeps on B4 risk timeouts and fail to beat the tri-basin portfolio.
- b4-truncation-and-bank-escapes.md — Structural truncation, homotopy, and envelope escapes on B4/bank seeds collapse or time out.
- surrogate-and-nae-escapes.md — Batched quadratic surrogate steps and NAE-from-scratch basins fall back to the floor.
- toroidal-axis-contraction.md — Orthogonal n-axis, combined (m,n), m-gated n-axis, or `exp(-q|n|)` toroidal scaling disrupts geometry.
- spectral-and-depth-perturbations.md — Mid-m bumps, asymmetric base splits, and custom LF-gated loops plateau or timeout.
- b1-power-law-contraction.md — Nonlinear power-law profiles, additive high-m bands, and fabricated matrices plateau or crash.
- risoliao6-basin-extension.md — Extending the R/Z contraction sweep to structurally distinct bank seeds plateaus immediately.
- structural-axis-perturbations.md — Radial translation, independent NAE basins, and iterative coordinate descent fail or starve budget.
- per-row-rz-and-3d-joint-sweeps.md — Per-row R/Z gradients, angular twists, stage-split cr variations, and multi-round joint 3D grids regress or time out.
- r0-rescale.md — Uniform major-radius (R0) rescale pre-contraction regresses by breaking aspect/QI coupling.
- m-transfer.md — Post-composition low-m to high-m zero-sum curvature transfer yields no gain over the optimal two-stage composition.
- m-dependent-cr-cz-ratio.md — Making the cr/cz decoupling ratio a function of poloidal mode `m` destroys global aspect coordination.
- b6-nae-independent-pivot.md — Parameterized nfp=3 NAE seed sweeps lack baseline L and collapse to the incumbent.
- stellar_p2-s100-78100567.md — Float typo regressions, failed NAE pivots (nfp=2/4), uncontraction, mode grafting, and n-axis perturbations fail.
- stellar_p2-s203-38950787.md — R/Z-split depth, NAE pivots, and blends regress to typo floor.
- stellar_p2-s201-47725970.md — Typo floor regressions, nfp=4/NAE basin collapses, and saturated stage-1 `(b1, c1)` interior micro-sweeps fail.
- stellar_p2-s202-55112695.md — Silent typos, exhausted stage-1/2 grids, dynamic bank seed drops, and an nfp=2 NAE timeout trap the run.
- stellar_p2-s204-63425638.md — Dynamic bank seed contraction, NAE nfp=2 pivots, coordinate ascent, adaptive stochastic ascent, m=0 phase shifts, BDC scaling, and hardcoded approximations cap the isolated run at 0.5835.
- stellar_p2-s205-85293087.md — Isolated state traps, fabricated low-mode boundary matrices, bank seed contraction caps, axisymmetric perturbations, spectral frequency-reweighting, and nfp=2→3 rescaling fail to recover baseline L, capping run at 0.5783.
- stellar_p2-s206-7450085.md — Isolated state traps without hardcoded matrix: spectral shifts, n-axis tapering, blends, multi-seed portfolios, hardcoded (8,7) approximations, shear, and rotation all cap the run at 0.581.
- stellar_p2-s207-18745292.md — Independent basins from NAE/ellipse seeds lack baseline objective_L and fail to achieve feasibility (capped at -0.83 score).
- independent-basin-violation-descent.md — Structural contraction, coordinate-axis, SPSA, forward-probe, and physics-first constraint-targeted transforms on NAE seeds strictly increase constraint violations, failing to bridge the gap to feasibility.

## performance-analysis
- feasibility-tolerance-economics.md — the official rule (`_DEFAULT_RELATIVE_TOLERANCE = 0.01`, max of 5 normalized violations, aspect ratio always binding) and the ~0.92 score-per-feasibility exchange rate.
- provenance-and-independence.md — pre-seed-bank runs never reached feasibility (official 0.0); every scoring result is a ≤0.5% perturbation of a public submission.

## implementation-insights
- budget-discipline.md — Slow evals cause timeouts; cap portfolios ≤15 evals in one batched eval_many.
- anchor-floor-rejection-bug.md — NEVER hardcode `base_b`/raw parent/diagonal as fallback; use the exact off-diagonal incumbent.
- lf-selection-bug.md — LF landmine gates must not override the train-score winner; use LF only as a tie-break.
- hardcoded-boundary-bugs.md — Never fabricate, manually truncate, or typo-corrupt Fourier matrices; VMEC strictly requires spectrally condensed shapes.
- dead-nae-loop-bug.md — When dynamically probing NAE seeds or building variable-length portfolios, ensure seeds are correctly appended to the `eval_many` batch list to prevent dead code.
- batch-key-and-silent-typos.md — `eval_many` dictionaries must match hardcoded key schemas exactly. Avoid silent float typos in matrices.
- shallow-uncontraction-pitfall.md — Reducing depth to buy an "honest margin" collapses `objective_L` faster than it saves feasibility.
- none-metric-crash-bug.md — Always None-guard NAE/bank metrics before arithmetic in `_accept_key` to prevent `TypeError` crashes.
- index-based-symmetry-zeroing-bug.md — Always use a boolean zero-pattern mask, not hardcoded column indices, to enforce stellarator symmetry dynamically.
- violation-first-selection.md — In isolated states without the public seed bank, select independent NAE/ellipse candidates strictly by minimizing maximum constraint violation.

## new-ideas
- low-nfp-nae.md — Pivoting to an nfp=2 NAE seed to exploit the L ∝ A/Nfp scaling law for a category jump in score.
- untested-ascent-mechanisms.md — Absorbed analyst rounds (SPSA, DFO); mostly refuted/tried.
- analyst-stellar_p2-s105-26196944-2.md — Decision log @ 24 cands.
- analyst-stellar_p2-s105-72881323-*.md — Historical decision logs (3, 4, 5, 6, 8).
- analyst-stellar_p2-s105-26196944-3.md — in-run Opus analysis + decision log @ 36 cands.
- analyst-stellar_p2-s105-26196944-4.md — in-run Opus analysis + decision log @ 48 cands.
- analyst-stellar_p2-s105-26196944-5.md — in-run analysis + decision log @ 60 cands.
- analyst-stellar_p2-s100-78100567-*.md — Historical decision logs for s100 (1 through 6).
- analyst-stellar_p2-s203-38950787-*.md — Historical decision logs (1, 2).
- analyst-stellar_p2-s201-47725970-*.md — Historical decision logs (1, 2).
- analyst-stellar_p2-s202-55112695-*.md — Historical decision logs (1, 2).
- analyst-stellar_p2-s204-63425638-*.md — Historical decision logs (1, 2, 3).
- analyst-stellar_p2-s205-85293087-*.md — Historical decision logs (1, 2, 3).
- analyst-stellar_p2-s206-7450085-1.md — in-run analysis + decision log @ 12 cands (stellar_p2-s206-7450085).
- analyst-stellar_p2-s207-18745292-1.md — in-run analysis + decision log @ 12 cands (stellar_p2-s207-18745292).

## Open directions
1. **The contraction ladder is done as a scoring strategy.** It has ~0.0007 of tolerance left.
2. **The only directions that still matter raise L at LOW feasibility**: a genuinely different basin (nfp=2 NAE) or a global structural transform.
3. **Run B6-nae-independent.** It is the only branch that tests whether this harness can reach a feasible QI boundary without a public seed.
4. TRUST val→official within a lineage: private 0.6400 > val 0.6380 > train 0.6269.
- new-ideas/analyst-stellar_p2-s100-78100567-1.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s100-78100567-2.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s100-78100567-3.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s100-78100567-4.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s100-78100567-5.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s100-78100567-6.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s105-72881323-3.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s105-72881323-4.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s105-72881323-5.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s105-72881323-6.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s105-72881323-8.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s201-47725970-1.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s201-47725970-2.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s202-55112695-1.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s202-55112695-2.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s203-38950787-1.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s203-38950787-2.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s204-63425638-1.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s204-63425638-2.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s204-63425638-3.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s205-85293087-1.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s205-85293087-2.md — (restored by index guard: dropped in last review)
- new-ideas/analyst-stellar_p2-s205-85293087-3.md — (restored by index guard: dropped in last review)
- analyst-stellar_p2-s207-18745292-1.md — in-run analysis + decision log @ 12 cands (stellar_p2-s207-18745292)
- analyst-stellar_p2-s207-18745292-2.md — in-run analysis + decision log @ 24 cands (stellar_p2-s207-18745292)
