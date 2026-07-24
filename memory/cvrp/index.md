# Memory index

## Current best (private = the real objective)
- c0021 (run cvrp-s37-18214945): private 3-eval median 0.607 (spread 0.54-0.64, the TIGHTEST ever) — best private median recorded; needs 3 more evals to confirm vs c0035's 6-eval 0.658. Mechanism: replaced the O(n^2) per-call Python argpartition KNN build with a cached numpy meshgrid build → measurably more wall-clock for the C kernel on the c0037 lean base. NOT an algorithm change — an overhead removal; the C kernel converts every freed millisecond into quality.
- c0020 (run cvrp-s17-78412700): private median ~0.70 over 9 evals (spread 0.49-0.87!) — statistical TIE with c0035, different family: regime-dispatcher + cheap forward-KNN inter-route Or-opt pre-filter. X-n200 has hit 0.15-0.19 (best short-route results ever) AND 0.98: the mechanism can find a better basin but does so unreliably.
- c0035 (run cvrp-s3-45465035): private 6-eval median 0.658. Untouched C LNS kernel + pure-Python Or-opt(1-3)+intra-2-opt polish overlay.
- c0006 (run cvrp-s23-89572315): train -0.117, val -0.228. Budget-gated second C LNS call + removed degenerate 3-opt overlay. No private evals yet.
- c0037 (run cvrp-s31-5170619): private 3-eval median 0.684 (0.56/0.73/0.68) — ties the champions at 6s from a LEANER program, and best-known 30s result: val30 -0.161 vs seed -0.194. Stripped heavy swap/cross-exchange ALNS, single unified cheap inter-route Or-opt(1) overlay, freed time to C.
- c0000 (seed of cvrp-s29/s31/s37): private medians 0.663 (s29) / 0.674 (s31, spread 0.59-0.88). Two $6 continuation runs (s29, s31) both ended seed-best at 6s — the 6s frontier looks saturated for this lineage; the 30s axis (c0037) is where s31 moved.
- Single evals lie: train noise ~0.09, private spread 0.1-0.18. Trust only medians (performance-analysis/score-noise-and-gate.md).

## Open directions
- TOP PRIORITY — the short-route regime (many routes, ~5 customers each: X-n242-k48 on val, X-n200-k36 on private) is where everything is lost at once: worst val30 gap (0.38 while the other two hit ~0.00 at 30s), worst variance (X-n200 draws span 0.15-0.98), and the champion families are all weakest there. Mechanism leads: with ~5-customer routes, intra-route polish is nearly useless — gains live in inter-route exchanges and route-count/assignment decisions; the cheap forward-KNN inter-route Or-opt (c0020) was the single best short-route result ever (0.15) — extend THAT family (swaps, 2-exchanges, relocate chains between routes, all KNN-filtered and strictly-improving) rather than intra-route variants. A reliable 0.3 on X-n200/X-n242 beats any other available gain.
- Large-n is where the score lives: train = X-n101/110/125/186, val = X-n153+X-n176.
- Verdicts are REGIME-conditional: n/k (customers per route) decides mechanism payoff, not just n (see performance-analysis/instance-structure.md). Since 2026-07-23 val = X-n153/228/242 matching private n/k.
- X-n303 is consistently the worst instance. Diagnose why (time starvation? neighbor-list quality at scale?).
- The gate now median-of-3s any |train delta| <= 0.10 (val 0.15). Deltas inside that band are ties.
- Do NOT retry: single long C calls, inter-route 2-opt* overlays, C recreate replacement, SA removal/tuning, C kernel rewrites, multi-start orchestration, blind mutations, static SA phase splitting, sequential C intensification, K widening (C and Python), route-merging/consolidation, dual-anchor/edge-anchor expansions, pre-C polish, intra-route Or-opt, regime dispatching, fixing the net_fwd bug (it gives 0 gradient), multi-move ALNS loops in Python, new cheap post-C overlay moves, C budget slicing.

## successful-patterns
- python-oropt-polish-overlay.md — the winning pattern: cheap strictly-improving Python Or-opt overlay. Budget-fill and O(1) arrays are saturated.
- surgical-bug-fixes.md — fixing concrete named defects beats new ideas

## ineffective-approaches
- c-side/c-kernel-rewrites.md — full/partial C rewrites, ports, consolidations, and new C entry points: crash or regress
- c-side/c-local-search-modifications.md — adding/refactoring moves in C try_moves (incl. K widening): neutral to catastrophic
- c-side/persistent-c-search-state.md — single long C calls / preserved SA state: 10+ exact ties
- c-side/recreate-replacement.md — greedy insertion or demand-urgency tie-breaks: refuted
- c-side/sa-acceptance-and-parameter-tuning.md — SA/ruin/temperature/restart tuning: neutral
- python-side/c-budget-slicing-and-multi-restart.md — splitting C LNS budget into multiple calls/starvation
- python-side/cheap-post-c-overlays.md — adding new KNN/Python local search post-C overlays
- python-side/sweep-construction-and-dual-basins.md — alternative initial constructions or dual basins
- python-side/python-orchestration-tuning.md — slice/check/budget/reordering tuning: neutral-to-negative
- python-side/inter-route-2opt-star-overlay.md — inter-route Python overlay moves starve C: 25+ attempts, zero wins
- python-side/multi-restart-and-multi-start.md — multi-start/multi-phase/perturbation loops divide budget, triggers 27825 starvation
- python-side/regime-dispatcher.md — dispatching polish strategy by customers-per-route: exhausted
- python-side/structural-repair-and-relocation.md — route merge/consolidation / stranded customer repair / pre-C polish / ejection chains: refuted
- python-side/overlay-dlb-and-knn-microopts.md — DLB restructuring / KNN cache / unified loops: exhausted
- python-side/intra-route-oropt-overlay.md — intra-route Or-opt / 3-opt / Or-opt(4) overlays: exhausted (at 6s; see 30s note in new-ideas)
- python-side/expanded-oropt-and-knn-variants.md — dual-anchor/bidirectional Or-opt KNN expansions: zero private gradient
- python-side/route-set-crossover-overlay.md — route recombination from dual C calls: time-starvation
- blind-mutations-pretracing.md — 120+ untargeted mutations/knob jitter: exhausted

## implementation-insights
- c-failure-modes.md — how C changes die (compile, heap, silent no-output, regress)
- generation-parse-failures.md — candidates lost to unparseable output or silent truncation
- python-oropt-bugs.md — the _py_segment_shift `net` vs `net_fwd` and dead time-break bugs

## performance-analysis
- run-metrics.md — per-instance score anatomy, attractors, timing budget
- instance-structure.md — customers-per-route (n/k) decides which mechanisms pay; val rebuilt to match private regime
- score-noise-and-gate.md — measured noise (train/val/private), gate misfire modes, decision rules

## new-ideas
- (open) anytime generalization: accepted candidates now carry a 30s-val score — improving it without losing 6s-val means the algorithm converges well at all budgets, not just the sprint. First evidence: c0037 (s31) val30 -0.161 vs seed -0.194 while tying 6s private (0.684) — the lean-overlay family converges better at long budgets.
- unevaluated-variants.md — proposals that died before scoring, with retry priorities
- dual-construction-and-degenerate-3opt.md — died-in-generation proposals for dual C kernel seeding and removing dead 3-opt code
