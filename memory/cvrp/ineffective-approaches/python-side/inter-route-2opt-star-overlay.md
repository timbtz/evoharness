# Inter-route 2-opt / 2-opt* in the Python overlay

Adding inter-route tail-swap (2-opt*), inter-route 2-opt (prefix reversal), inter-route single-customer swap, or inter-route segment relocate to the Python overlay was tried 25+ times and NEVER beat the incumbent.

## How it was tried
- 2-opt* (tail exchange): ~20 attempts (e.g., c0027, c0036, c0038). Heavy/interleaved variants actively starved C LNS (c0027 -0.384). Final-polish variants found zero improving moves because the C kernel's `tails()` already leaves its output 2-opt*-optimal within KNN.
  - c0009 (cvrp-s11-66566581): Added full-neighborhood `_py_two_opt_star` with O(1) prefix-load arrays. Train -0.059, val -0.510.
  - c0008 (run cvrp-s17-78412700): Crashed with IndexError (line 894) due to faulty bounds checking.
  - c0006 (run cvrp-s19-83885116): Added `_py_inter_suffix_swap` (KNN-anchored suffix-swap). Train -0.1552, val -0.2114.
  - c0018 (run cvrp-s19-83885116): Added `_py_two_opt_star` cheap Python overlay move. Train -0.1364, val -0.2855.
  - c0020 (run cvrp-s19-83885116): Consolidated Or-opt + intra-2opt + swap + 2-opt* into `_py_consolidated_overlay`. Train -0.1547, val -0.2744.
- Inter-route 2-opt (prefix reversal): c0040 (-0.0894, RUN BEST), c0034 (-0.0906). c0040 produced a new train best but the delta is entirely within the noise band. 
- Inter-route single-customer swap: c0025 (-0.0918), c0037 (-0.0906), c0030 (-0.0918), c0058 (-0.0960). All exactly score-neutral.
- Inter-route segment relocate: c0014 (cvrp-s11-66566581): Train -0.122, val -0.817. Rejected. c0018 (cvrp-s11-66566581): empty code/no output.
- Inter-route cross-exchange: c0007r2 (run cvrp-s19-83885116): Added `_py_cross_exchange` (swap segments 1-2). Train -0.1427, val -0.2373.
- Failed/Regressive implementations: c0033 (UnboundLocalError), c0042 (NameError), c0046/c0048/c0049 (starved C LNS).

## Why it failed
- Redundant neighborhood: the C kernel already exhausts KNN-local inter-route relocations, swaps, and tail exchanges inside `try_moves`. 
- Time-starvation: any overlay taking >0.15s pushes instance time 4.65 -> 4.8-4.85s+, triggering the 27825 starvation attractor (see performance-analysis/run-metrics.md).

## Verdict
refuted. Do not add ANY inter-route moves (including suffix-swap, cross-exchange, tail exchange) to the Python overlay again.

## REOPENED under k-matched val (2026-07-23)
- c0020 (run cvrp-s17-78412700): lightweight inter-route Or-opt with forward-only KNN lookups as a pre-filter before the full overlay. val -0.230 -> -0.175; private 6-eval median 0.639.
- The 18+ refuted attempts were heavier dual-anchor variants judged under the old val (long-route-skewed) — the cost/benefit flips with a cheap enough move set on short-route instances.
- Verdict amended: heavy inter-route overlays remain refuted; CHEAP forward-KNN inter-route Or-opt is promising on short-route regimes. High variance (one 0.98 X-n200 eval) is the open problem.
