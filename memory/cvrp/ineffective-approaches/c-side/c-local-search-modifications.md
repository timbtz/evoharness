# Adding or refactoring moves inside the C try_moves

Extending the C local search's move set (extra Or-opt lengths/orientations, ejection chains, 4-opt, wider K) or refactoring it for speed was tried ~15 times: neutral at best inside eval noise, occasionally catastrophic — the kernel's neighborhood is effectively saturated for the train sizes.

## How it was tried
- Or-opt extensions: c0008 (Or-opt up to length 4 — no output), c0013 (-0.099, accepted, neutral), c0017 (semantic no-op refactor — see performance-analysis/score-noise-and-gate.md), c0019 (K=30 + extra insertion site, -0.097 train but val -0.84, rejected), c0031 (both-orientation insertions + 2-opt* + K=25, -0.203), c0032 (Or-opt vs segment-endpoint neighbors + regret-2 tweak, -0.118, accepted via val tie, never became parent).
- Intra-route 2-opt: c0004 (dedicated intra-2-opt pass, -0.098), c0025 (restructured same-route 2-opt gating, -0.096), c0045 (intra-2opt cleanup inside C flatten step, -0.0966) — all neutral; the granular 2-opt/2-opt* branch already covers this territory.
- Exotic moves: c0011/c0011r1 (Or-opt "ejection chain" 3-for-2 trade; compile-fixed to -0.098 — neutral), c0023 (4-opt EAX-style cross-exchange + randomized scan order, -6.75, worst score of the run).
- Perf refactor: c0016 (neighbor-list row-major restructure of try_moves, -0.0906, accepted — exact tie, i.e. no measurable speed dividend).
- c0017 (cvrp-s11-66566581): tried widening K to 25 for n>=200, died from parse error.
- c0003/c0018 (cvrp-s13-71671014): widened C kernel `K = n - 2 < 25 ? n - 2 : 25` (or guarded `n >= 200`). c0003 train -0.0158 / val -0.24, c0018 train -0.134 / val -0.375. Higher move density failed to yield gains and likely just disrupted the SA state.
- c0002 (run cvrp-s19-83885116): widened C LNS KNN parameter `K = 30` for `n >= 150`. Train -0.1752, val -0.4278. Disrupted SA trajectory via excessive candidate scanning.

## Why it failed
- The existing granular neighborhood (relocate/Or-opt-1..3/swap/2-opt/2-opt* over KNN with don't-look bits) already reaches the same local optima; added moves either never fire (identical costs) or interact badly with the SA trajectory.
- Every change here re-rolls the timing/RNG dice: observed deltas (+/-0.01..0.1) match the noise floor.
- The catastrophic cases (c0023) show inter-route surgery in C is far riskier than its Python-overlay twin, which merely wastes time.

## Verdict
exhausted. Do not add moves to or refactor try_moves expecting train gains. Only revisit if targeting n>=200 behavior specifically AND with repeated evals; a single good train score proves nothing.
