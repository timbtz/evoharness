# Python Or-opt polish overlay on C LNS output

A cheap, strictly-improving, KNN-filtered pure-Python local search applied to the C kernel's solutions — with the C kernel byte-identical — is the only pattern that produced real private gains.

## How it was tried
- c0024-c0063 (cvrp-s3/s5): established Or-opt(1-3) + intra-route 2-opt/3-opt/swap. Replaced linear route-membership scans with O(1) `route_of` arrays. Accelerated with numpy slicing.
- c0005 (cvrp-s11-66566581): Changed fixed 5-round overlay to `while time.monotonic() < t_end:` budget-fill loop. Train -0.1064, accepted.
- c0003/c0004 (cvrp-s11-66566581): Added intra-route Or-opt and single-greedy passes (rejected for noise, but safe).
- c0024 (cvrp-s13-71671014): Added budget-filled `_py_segment_shift` (inter-route segment shift with full O(n) insertion scan for segments 1..4) alongside the existing Or-opt overlay. Accepted as new run best with train -0.0219 / val -0.1732.

## Why it worked
- Zero risk to the proven C kernel (every C-touching family crashed or regressed).
- The overlay only fires on strictly improving moves and is validated (`_valid`).
- Cheapness is load-bearing: the light overlay costs ~0.01-0.04s/instance.
- Real payoff is on LARGE instances: private mean gap dropped to 0.581% because Or-opt polish scales to n=200-303.
- Full-scan insertion (as in c0024's `_py_segment_shift`) finds globally optimal relocations that KNN-anchored Or-opt misses, pushing the score forward without hitting starvation.

## Verdict
exhausted for micro-optimizations. Replacing linear route lookups with O(1) arrays is fully saturated (tried 30+ times). Do not propose O(1) routing array fixes, budget-fill loop changes, or numpy slicing again.
