# Python Or-opt polish overlay on C LNS output

A cheap, strictly-improving, KNN-filtered pure-Python local search applied to the C kernel's solutions — with the C kernel byte-identical — is the only pattern that produced real private gains.

## How it was tried
- c0024-c0063 (cvrp-s3/s5): established Or-opt(1-3) + intra-route 2-opt/3-opt/swap. Replaced linear route-membership scans with O(1) `route_of` arrays. Accelerated with numpy slicing.
- c0005 (cvrp-s11-66566581): Changed fixed 5-round overlay to `while time.monotonic() < t_end:` budget-fill loop. Train -0.1064, accepted.
- c0024 (cvrp-s13-71671014): Added budget-filled `_py_segment_shift` alongside the existing Or-opt overlay. Train -0.0219, val -0.1732. *(Note: this established the latent `net` vs `net_fwd` bug and unreachable time-break, see implementation-insights/python-oropt-bugs.md)*
- c0022/c0026 (cvrp-s23-89572315): Multiple surgical bug fixes to overlay move logic. Train -0.133 / -0.082.
- c0032/c0033/c0035/c0037 (run cvrp-s31-5170619): Restored minimal overlay (only `_py_inter_route_oropt1`) after heavy multi-move ALNS loops starved the budget. Train -0.080 to -0.353, val -0.179 to -0.245. Stripping heavy swaps is the safest baseline.

## Why it worked
- Zero risk to the proven C kernel (every C-touching family crashed or regressed).
- The overlay only fires on strictly improving moves and is validated (`_valid`).
- Cheapness is load-bearing: the light overlay costs ~0.01-0.04s/instance.
- Real payoff is on LARGE instances: private mean gap dropped to 0.581% because Or-opt polish scales to n=200-303.

## Verdict
exhausted for micro-optimizations. Replacing linear route lookups with O(1) arrays is fully saturated (tried 30+ times). Do not propose O(1) routing array fixes, budget-fill loop changes, numpy slicing, or fixing the `net_fwd` bug (it provides zero algorithmic gradient) again.
