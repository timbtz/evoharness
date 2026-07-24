# Cheap strictly-improving Python post-C overlays (New move types)

Adding new inter-route or intra-route strictly-improving Python overlays (Or-opt(2-3), inter-route 1-1 swaps, 2-opt*, intra-route 2-opt/Or-opt) to polish the C kernel's final solution provides zero algorithmic gradient. The C kernel already exhausts these KNN-filtered move classes internally.

## How it was tried
- c0005 (run cvrp-s37-18214945): Added KNN-filtered inter-route Or-opt(2-3) segment relocate. Train -0.242, val -0.176.
- c0006 (run cvrp-s37-18214945): Added cheap intra-route 2-opt post-polish (reversing full route subsequences). Train -0.146, val -0.301.
- c0009 (run cvrp-s37-18214945): Added KNN-filtered inter-route single-customer SWAP (1-1) bounded final polish. Train -0.187, val -0.274.
- c0012 (run cvrp-s37-18214945): Pure-Python intra-route 2-opt applied to final best, explicitly replacing the Or-opt(1) overlay. Train -0.594, val -0.274.
- c0016 (run cvrp-s37-18214945): Added Python inter-route 2-opt* (edge exchange between tail ends). Train -0.211, val -0.213.
- c0017 (run cvrp-s37-18214945): Added bidirectional inter-route Or-opt(1) pass (relocating `u` to `c`'s route). Train -0.284, val -0.235.
- c0018 (run cvrp-s37-18214945): Added KNN-filtered intra-route Or-opt(1,2,3) post-polish. Train -0.270, val -0.235.
- c0019 (run cvrp-s37-18214945): Added KNN-filtered inter-route Or-opt(2) overlay. Train -0.312, val -0.255.
- c0020 (run cvrp-s37-18214945): Added KNN-filtered inter-route 1-1 swap bounded post-polish. Train -0.309, val -0.280.

## Why it failed
- The C kernel already exhausts KNN-local inter-route relocations, 1-1 swaps, 2-opt, 2-opt*, and Or-opt(1-3) inside `try_moves`. Any residual gains from Python overlays are strictly below the eval noise band.
- Inter-route or intra-route full-scan O(n²) Python passes miss their time budget, causing X-n101 to hit the 27825 starvation attractor.
- Even when kept extremely cheap and bounded (c0009, c0019), they provide zero measurable algorithmic payoff.

## Verdict
exhausted. Do not add ANY new inter-route or intra-route moves (Or-opt(2-3), 1-1 swaps, intra-2-opt, 2-opt*) to the Python overlay.
