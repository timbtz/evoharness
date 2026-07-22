# Memory index

## Current best
- c0035, train -0.0894, run cvrp-s3-45465035, private mean gap 0.581% on X-n200/251/303 — best private result across all runs (earlier runs: 0.70-0.80%).
- c0035 = untouched C LNS kernel + pure-Python Or-opt(1-3)+intra-2-opt polish overlay (see successful-patterns/python-oropt-polish-overlay.md).
- CAUTION: single-eval train deltas below ~0.1 are noise (see performance-analysis/score-noise-and-gate.md). c0035's train edge over the -0.0906 cluster (55688 vs 55690 on X-n125) is within noise; its private result is the real evidence.

## Open directions
- Persistent C search state / one long C call instead of 1.5s slices: strongest untested idea, 0 evaluations ever — see new-ideas/persistent-c-search-state.md.
- Train instances are n<=125 with instances 1-2 at optimum; private is n=200-303, so changes that only help large instances are invisible on train. Target large-n behavior explicitly (see performance-analysis/run-metrics.md).
- The -0.0906 plateau is a single attractor (X-n125 cost 55690); beating it requires a reproducible move below 55690, not one lucky run.
- Do NOT retry: inter-route 2-opt*/2-opt overlays (18 attempts), C recreate replacement (2x refuted), SA removal/tuning (refuted), C kernel rewrites (8x refuted), blind mutations (120+ pre-tracing candidates).

## successful-patterns
- python-oropt-polish-overlay.md — the winning pattern: cheap, strictly-improving Python Or-opt polish on C output; exact mechanism and why heavier overlays backfire
- surgical-bug-fixes.md — fixing concrete bugs in existing code (savings orientation, compile repairs) outperforms new ideas

## ineffective-approaches
- inter-route-2opt-star-overlay.md — 18 attempts to add 2-opt*/inter-2-opt to the Python overlay; zero wins; two distinct failure modes
- c-kernel-rewrites.md — full/partial C rewrites, ports, single-call consolidations: all crashed or regressed heavily
- recreate-replacement.md — swapping regret-2 recreate for faster greedy insertion: refuted twice
- sa-acceptance-and-parameter-tuning.md — removing/tightening SA and ruin/temperature/restart tuning: refuted to neutral
- c-local-search-modifications.md — adding/refactoring moves inside C try_moves (Or-opt variants, ejection chain, 4-opt, K changes): neutral at best, sometimes catastrophic
- diversification-perturbations.md — double-bridge, randomized recreate/scan-order/restarts: score-neutral
- python-orchestration-tuning.md — slice sizes, check frequency, budget shuffling: neutral-to-negative
- blind-mutations-pretracing.md — 120+ pre-tracing candidates of untargeted mutations/knob jitter: exhausted

## implementation-insights
- c-failure-modes.md — how C changes die (compile errors, heap corruption, silent no-output, runs-but-regresses) incl. pre-tracing recurrences; what a repair step can and cannot fix
- generation-parse-failures.md — candidates lost to unparseable output; parse-failed ideas are open, not refuted

## performance-analysis
- run-metrics.md — where the score lives: per-instance anatomy, the 27825/55690 attractors, timing budget, per-run history
- score-noise-and-gate.md — eval noise is ~0.1 train / ~0.14 val per single run; how the holdout gate misfires both ways

## new-ideas
- persistent-c-search-state.md — one long C LNS call preserving SA state (c0054/c0037/c0009): mechanism-backed, never evaluated
- unevaluated-variants.md — small proposals that died before scoring (c0003 SA-acceptance check, c0008 Or-opt-4, c0010 demand-urgency recreate), with retry priorities
