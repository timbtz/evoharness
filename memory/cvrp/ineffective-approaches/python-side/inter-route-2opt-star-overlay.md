# Inter-route 2-opt / 2-opt* in the Python overlay

Adding inter-route tail-swap (2-opt*), inter-route 2-opt (prefix reversal), inter-route single-customer swap, or inter-route segment relocate to the Python overlay was tried 30+ times and NEVER beat the incumbent.

## How it was tried
- 2-opt* (tail exchange): ~20 attempts. Heavy/interleaved variants actively starved C LNS (c0027 -0.384). Final-polish variants found zero improving moves because the C kernel's `tails()` already leaves its output 2-opt*-optimal within KNN.
  - c0012 (run cvrp-s31-5170619): Added cheap `_py_two_opt_star` using mutual-KNN edge pairs with O(1) prefix-load capacity checks. Train -0.094, val -0.232. Rejected.
  - c0013 (run cvrp-s31-5170619): Added `_py_two_opt_star_short` gated to short-route regime (≤8 customers/route). Train -0.132, val -0.299. Rejected.
  - c0021r2 (run cvrp-s31-5170619): Full `_py_tail_exchange` using prefix-load tracking. Train -0.157, val -0.239. Rejected.
- Inter-route 2-opt (prefix reversal): c0040 (-0.0894, RUN BEST), c0034 (-0.0906). c0040 produced a new train best but the delta is entirely within the noise band. 
- Inter-route single-customer swap: c0025 (-0.0918), c0037 (-0.0906), c0030 (-0.0918), c0058 (-0.0960), c0017 (cvrp-s31-5170619: -0.137). All exactly score-neutral.
  - c0031/c0033/c0034/c0035/c0037 (run cvrp-s31-5170619): `_inter_route_swap` added as part of ALNS or isolated. All caused train regressions (-0.129 to -0.352) by starving the C kernel.
- Inter-route segment relocate / cross-exchange: c0014 (cvrp-s11-66566581): -0.122. 
  - c0009r1 (run cvrp-s31-5170619): Added `_py_2exchange` (swap suffix-tail segments). Train -0.075 (tied), but val -0.719 (catastrophic regression).
  - c0010 (run cvrp-s31-5170619): Added `_py_cross_exchange_2` (swap 2-customer segments). Train -0.107, val -0.274.
  - c0027r1 (run cvrp-s31-5170619): `_py_cross_exchange` swapping interior segments. Train -0.145, val -0.342. Rejected.

## Why it failed
- Redundant neighborhood: the C kernel already exhausts KNN-local inter-route relocations, swaps, and tail exchanges inside `try_moves`. 
- Time-starvation: any overlay taking >0.15s pushes instance time 4.65 -> 4.8-4.85s+, triggering the 27825 starvation attractor (see performance-analysis/run-metrics.md).
- c0009r1 proves that suffix-tail exchanges in Python can cause massive private/val generalization failures even when the train score looks like a tie.

## Verdict
refuted. Do not add ANY inter-route moves (including suffix-swap, cross-exchange, tail exchange, KNN 2-opt*, or 1-1 swaps) to the Python overlay again.

## REOPENED under k-matched val (2026-07-23)
- c0020 (run cvrp-s17-78412700): lightweight inter-route Or-opt with forward-only KNN lookups as a pre-filter before the full overlay. val -0.230 -> -0.175; private 6-eval median 0.639.
- The 18+ refuted attempts were heavier dual-anchor variants judged under the old val (long-route-skewed) — the cost/benefit flips with a cheap enough move set on short-route instances.
- Verdict amended: heavy inter-route overlays remain refuted; CHEAP forward-KNN inter-route Or-opt is promising on short-route regimes. High variance (one 0.98 X-n200 eval) is the open problem.
