# Intra-route Or-opt, 3-opt, and extended Or-opt(4) in the Python overlay

Adding intra-route segment relocations (Or-opt), 3-opt, or extending the cross-route Or-opt to segment length 4 in the Python overlay fails to improve the score because the C kernel and existing 2-opt polish already reach these local optima.

## How it was tried
- c0022 (cvrp-s11-66566581): added pure-Python intra-route Or-opt (segments 1-3). Train -0.051, val -0.481.
- c0021r1 (cvrp-s11-66566581): numpy-vectorized intra-route Or-opt(1-3). Train -0.106, val -0.423.
- c0019r1 (cvrp-s11-66566581): extended existing Or-opt loop to length 4. Train -0.111, val -0.577.
- c0024 (cvrp-s11-66566581): full O(n²) intra-2opt pass for small/medium n combined with intra-route Or-opt. Train -0.119, val -0.409.
- c0023 (cvrp-s13-71671014): added `try_intra_or_opt` to overlay. Train -0.073, val -0.358.
- c0017 (run cvrp-s19-83885116): added `_py_intra_2opt_dlb`. Train -0.1195, val -0.2705.
- c0019 (run cvrp-s19-83885116): added `_py_intra_oropt` with explicit reversal variants. Train -0.0820, val -0.2957.
- c0001, c0004, c0011, c0016 (run cvrp-s29-99479842): Attempted various full-scan or KNN-filtered intra-route Or-opt(1-3). Train -0.071 to -0.106.
- c0014 (run cvrp-s31-5170619): Added pure-Python `_py_intra_3opt` full-scan segment reversal. Train -0.098, val -0.233.

## Why it failed
- The C kernel's granular local search already applies intra-route 2-opt/3-opt exhaustively. Any residual gains from intra-route segment shifts are strictly below the eval noise band.
- Adding segment length 4 does not close any meaningful gap left by lengths 1-3.

## Verdict
exhausted. Do not add intra-route Or-opt, 2-opt DLB, Or-opt(4), or intra-route 3-opt to the Python overlay.
