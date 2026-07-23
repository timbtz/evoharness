# Intra-route Or-opt and extended Or-opt(4) in the Python overlay

Adding intra-route segment relocations (Or-opt) or extending the cross-route Or-opt to segment length 4 in the Python overlay fails to improve the score because the C kernel and existing 2-opt/3-opt polish already reach these local optima.

## How it was tried
- c0022 (cvrp-s11-66566581): added pure-Python intra-route Or-opt (segments 1-3) sweep to the overlay. Train -0.051, val -0.481. Rejected.
- c0021r1 (cvrp-s11-66566581): added numpy-vectorized intra-route Or-opt(1-3) to the overlay. Train -0.106, val -0.423. Rejected (exact tie).
- c0019r1 (cvrp-s11-66566581): extended existing Or-opt loop to length 4. Train -0.111, val -0.577. Rejected.
- c0024 (cvrp-s11-66566581): full O(n²) intra-2opt pass for small/medium n combined with intra-route Or-opt. Train -0.119, val -0.409. Rejected.
- c0016r1 (cvrp-s11-66566581): attempted Or-opt(4) with dual-anchor, died to parse error (truncated code).
- c0023 (cvrp-s13-71671014): added intra-route Or-opt(1-3) (`try_intra_or_opt`) to overlay and reverted c0001's 3-phase C LNS split back to single-call. Train -0.073, val -0.358. Rejected for train regression.
- c0003, c0009, c0014, c0015, c0016 (run cvrp-s17-78412700): Added various Python intra-route Or-opt loops (cyclic shifts, full-scan single/multi-node relocation). Train scored -0.0950 to -0.1593, val -0.264 to -0.301. All rejected.
- c0017 (run cvrp-s19-83885116): added `_py_intra_2opt_dlb` with KNN-filtered intra-route 2-opt and don't-look bits. Train -0.1195, val -0.2705. Rejected.
- c0019 (run cvrp-s19-83885116): added `_py_intra_oropt` with explicit reversal variants. Train -0.0820, val -0.2957. Rejected.

## Why it failed
- The C kernel's granular local search already applies intra-route 2-opt/3-opt exhaustively. Any residual gains from intra-route segment shifts are strictly below the eval noise band.
- The writer for c0023 incorrectly hypothesized that the C kernel skips intra-route Or-opt. While unverified by the code here, the eval results confirm it provides zero benefit on train instances.
- Adding segment length 4 does not close any meaningful gap left by lengths 1-3.

## Verdict
exhausted. Do not add intra-route Or-opt, 2-opt DLB, or Or-opt(4) to the Python overlay.
