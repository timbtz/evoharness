# Memory index — stellar_p2 (ConStellaration P2, simple-to-build QI stellarator)

NOTE TO REFINERS: plain reverts to the incumbent are near-wasted slots — when the writer's idea is broken, either REPAIR its mechanism so the idea actually runs, or propose YOUR OWN distinct improvement grounded in Open directions.

Claim ladder (official high-fid scores; sources: arXiv 2506.19583, arXiv 2502.12845, HF leaderboard 2026-07-23): beat ALM baseline 0.431 (34h x 96 vCPU) -> beat ExLLM 0.505 -> approach leaderboard #1 0.6361 (davidkh).
Truth = official evaluate() only; train/val shaped scores are proxies.

## successful-patterns
- baseline-alm-tricks.md — the 0.431 recipe: spectrum scaling 1.5, ALM schedules, QI log-transform, NAE seeding
- batch-population-and-coordinate-moves.md — structured budget spending + coordinate-deflation + surgical inflation tuning (c0009 -> c0028, c0025)
- quadrupole-shear-and-dilation.md — zero-RNG deterministic geometric transforms to break mirror ratio and aspect ratio plateaus
- margin-aware-polish.md — feasibility-margin-aware polish on bank seeds prevents val collapse (0.6286 val)

## ineffective-approaches
- inner-update-rule-evolution.md — don't reinvent CMA-es / ALM gradients
- heuristic-constraint-biasing.md — hardcoded directional mutations or roulettes to fix constraints destroy physics geometry
- schedule-rewrites.md — replacing the 4-phase schedule crashes the score
- seed-and-schedule-tuning.md — widening elite pools, sigma tweaks break exploration
- seed-portfolio-bloat.md — expanding seed budget or mixing ellipse geometries
- coordinate-probes-and-sweep-rewrites.md — bidirectional/axis sweeps or magnitude-ranked sweeps destroy budget trajectory
- alternating-sign-sweeps.md — alternating +/- signs per sweep wrap never fires
- seed-portfolio-and-bias-rewrites.md — changing NAE portfolio parameters, swapping unmetered seeds, or injecting ALM heuristic penalties
- worst-member-interpolation.md — blending worst elites with best destroys geometric stability

## implementation-insights
- vlf-blindness-landmines.md — identical-to-parent train score after a change = UNTESTED, not safe (c0001 landmine)
- refiner-guidance.md — refiners: repair or propose, never plain-revert
- numpy-shape-bugs.md — variable-shape matrices kill vectorization; pad/project before arithmetic
- risky-seed-modifications.md — unsafe unpacking of fm.eval and in-place array mutations trigger pydantic crashes
- deterministic-plateau-and-decorations.md — exact score ties mean dead code paths or unfiring triggers

## performance-analysis
- seed-bank-regime.md — THE CURRENT GAME: 12 public seeds, bar = 0.636, eval frugality, recombination directions
- fidelity-dial.md — 1.4s vlf / 2.2s lf / ~64-128s official; disagreement = red flag
- seed-baseline.md — raw NAE -0.95..-1.08, seed ends -0.665; QI is the wall

## new-ideas
- crash-classifier-guards.md — react to fm.last_error classes; pre-eval screens
- dataset-seed-portfolios.md — HF-dataset seeds (harness policy decision)
- vmec-hot-restart.md — 2-5x eval savings, blocked on fm API decision
- candidate-fitted-surrogates.md — local surrogates; scores never leave the candidate
- early-recombination.md — mean-crossover of pool[0]/[1] or bank seeds; refuted
- free-seed-slot-extremes.md — replacing duplicate portfolio specs with mirror-crushing extremes
- ngopt-endgame-slice.md — NGOpt on coarse modes at Phase 4; refuted at various budgets

## Current best
c0001f of run stellar_p2-s17-78763752: train 0.6197 (val 0.6286). Polished the top bank seed with feasibility-margin-aware selection (penalty weight 8.0, SAFE limit 0.006) to avoid tolerance camping. Official private 0.0 — nothing feasible yet.

## Open directions
1. Tune `SAFE` (0.006), `SAFE_RET` (0.0075), and `QI_SAFE` (-4.05) dynamically in the margin-aware polish.
2. Close ONE constraint fully without regressing the others, then QI last.
3. Find new early-basin seed parameter extremes (like c0025/c0026f) without duplicating specs.
4. Keep L high while repairing.
