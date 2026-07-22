# Python Or-opt polish overlay on C LNS output

A cheap, strictly-improving, KNN-filtered pure-Python local search (Or-opt segments 1-3 + intra-route 2-opt) applied to the C kernel's solutions — with the C kernel byte-identical — is the only pattern that produced the run best.

## How it was tried
- c0024 (accepted, -0.0906): added `_py_or_opt1` (single-customer relocate, KNN-filtered, best-improvement, `try/finally`-guarded) called once per outer-loop iteration inside a `time < deadline-0.2` guard. Score-neutral on train but established the safe overlay skeleton.
- c0035 (accepted, -0.0894, RUN BEST, private 0.581%): replaced `_py_or_opt1`'s body with Or-opt segment sizes 1-3 (both orientations) + intra-route 2-opt, still KNN-filtered and strictly improving; added a second call as a final polish on `best` before returning. Solver time stayed 4.66s (overlay is cheap).
- c0044 (accepted, -0.0918) and c0051 (accepted, -0.0906): careful augmentations (prefix-sum capacity checks, bounds-check hardening) that preserved the pattern — score-neutral, confirming the overlay is saturated with the current move set.

## Why it worked
- Zero risk to the proven C kernel (every C-touching family crashed or regressed; see ineffective-approaches/c-kernel-rewrites.md and c-local-search-modifications.md).
- The overlay only fires on strictly improving moves and is validated (`_valid`) before replacing `best`, so worst case is a no-op.
- Cheapness is the load-bearing property: the light overlay costs ~0.01s/instance. Every heavier overlay (inter-2-opt* variants) pushed instance time 4.65->4.85s and starved the C LNS (see ineffective-approaches/inter-route-2opt-star-overlay.md).
- Real payoff is on LARGE instances: train gain was noise-level (55690->55688 on X-n125), but private mean gap dropped from ~0.70-0.80% (earlier runs) to 0.581% — Or-opt 1-3 polish scales to n=200-303 where the C LNS converges less fully.

## Verdict
promising but near-saturated. The move set (Or-opt 1-3, intra-2-opt) is done; 18 attempts to extend it with inter-route moves all failed. A future writer should not add more overlay moves — instead find changes that specifically help n>=200 (invisible on train; see performance-analysis/run-metrics.md).
