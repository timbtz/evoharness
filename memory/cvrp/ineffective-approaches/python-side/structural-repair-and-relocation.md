# Structural repair, route merging, consolidation, and stranded customer relocation

Applying structural repair passes (merging near-empty routes, consolidating to save depot trips, relocating stranded customers, or pre-C polishing) as a Python overlay to fix route fragmentation.

## How it was tried
- c0010 (cvrp-s11-66566581): added `_merge_routes` post-LNS pass. Train -0.120, val -0.423.
- c0012 (run cvrp-s19-83885116): added `_py_greedy_route_merge` step. Train -0.0957, val -0.2933.
- c0013 (run cvrp-s19-83885116): added `_py_route_merge` polish step KNN-filtered. Train -0.1013, val -0.3218.
- c0020 (run cvrp-s29-99479842): Ran the overlay BEFORE the C LNS call for `n>=150`. Train -0.1066, val -0.4750.
- c0002 (run cvrp-s31-5170619): Added `_py_relocate_fill` KNN-filtered pass moving stranded customers. Train -0.098, val -0.320.
- c0019 (run cvrp-s31-5170619): Added `_py_route_consolidate` to try dissolving the smallest-load routes via best-insertion. Train -0.118, val -0.304.
- c0024r1 (run cvrp-s31-5170619): Added `_py_ejection_oropt1` to relocate blocked customers via ejection chains. Train -0.157, val -0.270.

## Why it failed
- Redundant with the C kernel: the LNS ruin-and-recreate naturally manages route count and consolidates globally better insertions. 
- Route merges and customer relocations saving depot trips are functionally similar to Or-opt or 2-opt* cross-route relocations, which the C kernel already exhausts locally. 
- Pre-C polishing is redundant because C's internal `local_search` immediately destroys the structural bias of the initial savings construction anyway.

## Verdict
exhausted. Do not add structural repair, route merge/consolidation passes, stranded customer relocation, ejection chains, or pre-C polish to the Python overlay.
