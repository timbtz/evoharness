# Expanded Or-opt and KNN variants in the Python overlay

Modifying the Or-opt overlay to use dual-anchors (searching neighbors of both segment endpoints), removal-side endpoints, or otherwise expanding the candidate neighbor sets fails to improve over the standard Or-opt overlay.

## How it was tried
- c0007 (run cvrp-s17-78412700): Added `_py_xoropt_expanded` checking insertion routes via `union(anchor KNN ∪ removal-side endpoint KNN)`. Train -0.0754, val -0.2338. Rejected (tie).
- c0011 (run cvrp-s17-78412700): Re-implemented a dual-anchor Or-opt using `anchors = [seg[0], seg[-1]]` inside `_py_or_opt1`. Train -0.1205, val -0.2732. Rejected (tie).
- c0020 (run cvrp-s17-78412700): Added `_py_fast_oropt` (ultra-fast forward-KNN Or-opt 1,2). Train -0.0482, val -0.1751. Accepted as tie.
- c0010 (run cvrp-s19-83885116): added `_py_edge_anchor_sweep` to examine KNN neighbors of all edge endpoints (structurally distinct traversal of standard moves). Train -0.0775, val -0.2637. Rejected.
- c0016 (run cvrp-s19-83885116): combined tier 1 and tier 2 into a single unified loop passing shared KNN. Train -0.1427, val -0.2831. Rejected.

## Why it failed
- The standard Python overlay and the C kernel already exhaustively search the K=15-20 nearest neighbors of segment endpoints. Expanding the candidate set introduces minor overhead (checking duplicate routes, longer loops) for zero algorithmic payoff on train/val instances.
- The mechanism hypothesis (finding non-KNN-local optima the C kernel misses) was refuted: either the C LNS trajectory already compensates for KNN locality via SA ruin-and-recreate, or such moves simply do not exist in high-density optimal route regions.

## Verdict
exhausted. Do not add expanded candidate lists, dual-anchor variants, edge-anchor sweeps, or bidirectional KNN expansions to the Python Or-opt overlay.
