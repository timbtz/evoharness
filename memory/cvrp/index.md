# Memory index

## Current best (private = the real objective)
- c0020 (run cvrp-s17-78412700): private median ~0.70 over 9 evals (spread 0.49-0.87!) — statistical TIE with c0035, different family: regime-dispatcher + cheap forward-KNN inter-route Or-opt pre-filter. X-n200 has hit 0.15-0.19 (best short-route results ever) AND 0.98: the mechanism can find a better basin but does so unreliably.
- c0035 (run cvrp-s3-45465035): private 6-eval median 0.658. Untouched C LNS kernel + pure-Python Or-opt(1-3)+intra-2-opt polish overlay.
- c0006 (run cvrp-s23-89572315): train -0.117, val -0.228. Budget-gated second C LNS call + removed degenerate 3-opt overlay. No private evals yet.
- Single evals lie: train noise ~0.09, private spread 0.1-0.18. Trust only medians (performance-analysis/score-noise-and-gate.md).

## Open directions
- TOP PRIORITY — variance reduction: c0020's private draws span 0.49-0.87. The search sometimes lands in a far better basin (X-n200 0.15!) by luck. Make that reliable: e.g., detect a bad trajectory early (cost threshold at 40% budget) and restart; or make the initial construction less luck-dependent. A candidate that turns 0.15 from a lucky draw into the typical draw wins everything.
- Large-n is where the score lives: train = X-n101/110/125/186, val = X-n153+X-n176.
- Verdicts are REGIME-conditional: n/k (customers per route) decides mechanism payoff, not just n (see performance-analysis/instance-structure.md). Since 2026-07-23 val = X-n153/228/242 matching private n/k.
- X-n303 is consistently the worst instance. Diagnose why (time starvation? neighbor-list quality at scale?).
- The gate now median-of-3s any |train delta| <= 0.10 (val 0.15). Deltas inside that band are ties.
- Do NOT retry: single long C calls, inter-route 2-opt* overlays, C recreate replacement, SA removal/tuning, C kernel rewrites, multi-start orchestration, blind mutations, static SA phase splitting, sequential C intensification, K widening (C and Python), route-merging, dual-anchor/edge-anchor expansions.

## successful-patterns
- python-oropt-polish-overlay.md — the winning pattern: cheap strictly-improving Python Or-opt overlay. Budget-fill and O(1) arrays are saturated.
- surgical-bug-fixes.md — fixing concrete named defects beats new ideas

## ineffective-approaches
- c-side/c-kernel-rewrites.md — full/partial C rewrites, ports, consolidations, and new C entry points: crash or regress
- c-side/c-local-search-modifications.md — adding/refactoring moves in C try_moves (incl. K widening): neutral to catastrophic
- c-side/persistent-c-search-state.md — single long C calls / preserved SA state: 10+ exact ties
- c-side/recreate-replacement.md — greedy insertion or demand-urgency tie-breaks: refuted
- c-side/sa-acceptance-and-parameter-tuning.md — SA/ruin/temperature/restart tuning: neutral
- python-side/inter-route-2opt-star-overlay.md — inter-route Python overlay moves starve C: 25+ attempts, zero wins
- python-side/multi-restart-and-multi-start.md — multi-start/multi-phase divides budget, triggers 27825 starvation
- python-side/single-call-restoration.md — reverting multi-phase parents to single C LNS: 13 ties across runs
- python-side/python-orchestration-tuning.md — slice/check/budget tuning: neutral-to-negative
- python-side/diversification-perturbations.md — double-bridge, randomized restarts: score-neutral
- python-side/structural-repair-overlay.md — route merge / stranded customer repair: refuted
- python-side/overlay-dlb-and-knn-microopts.md — DLB restructuring / KNN cache / KNN vectorization: exhausted
- python-side/intra-route-oropt-overlay.md — intra-route Or-opt / Or-opt(4) overlays: exhausted
- python-side/expanded-oropt-and-knn-variants.md — dual-anchor/bidirectional Or-opt KNN expansions: zero private gradient
- python-side/route-set-crossover-overlay.md — route recombination from dual C calls: time-starvation
- blind-mutations-pretracing.md — 120+ untargeted mutations/knob jitter: exhausted

## implementation-insights
- c-failure-modes.md — how C changes die (compile, heap, silent no-output, regress)
- generation-parse-failures.md — candidates lost to unparseable output or silent truncation

## performance-analysis
- run-metrics.md — per-instance score anatomy, attractors, timing budget
- instance-structure.md — customers-per-route (n/k) decides which mechanisms pay; val rebuilt to match private regime
- score-noise-and-gate.md — measured noise (train/val/private), gate misfire modes, decision rules

## new-ideas
- (open) anytime generalization: accepted candidates now carry a 30s-val score — improving it without losing 6s-val means the algorithm converges well at all budgets, not just the sprint
- regime-dispatcher.md — TOP PRIORITY untested idea: dispatch polish strategy by estimated customers-per-route (both blocks already proven in different regimes)
- unevaluated-variants.md — proposals that died before scoring, with retry priorities
- dual-construction-and-degenerate-3opt.md — died-in-generation proposals for dual C kernel seeding and removing dead 3-opt code
