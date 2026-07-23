# Structural repair overlays (route merging and stranded customers)

Applying structural repair passes (merging near-empty routes or relocating stranded customers) as a Python post-LNS overlay to fix route fragmentation.

## How it was tried
- c0010 (cvrp-s11-66566581): added `_merge_routes` post-LNS pass to consolidate routes by saving a depot round-trip when capacity allows (4 concatenation variants). Train -0.120, val -0.423. Rejected.
- c0012 (run cvrp-s19-83885116): added `_py_greedy_route_merge` step trying all 4 endpoint orientations. Train -0.0957, val -0.2933. Rejected.
- c0013 (run cvrp-s19-83885116): added `_py_route_merge` polish step KNN-filtered with capacity check. Train -0.1013, val -0.3218. Rejected.

## Why it failed
- Redundant with the C kernel: the LNS ruin-and-recreate naturally manages route count and consolidates globally better insertions. 
- Route merges saving a depot round-trip are functionally similar to Or-opt or 2-opt* cross-route relocations, which the C kernel already exhausts locally. 

## Verdict
exhausted. Do not add structural repair or route merge passes to the Python overlay.
