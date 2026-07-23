# Python overlay DLB fixes, KNN caching, and KNN vectorization (micro-optimizations)

Refactoring the Python Or-opt overlay's don't-look-bits (DLB), caching KNN arrays, or vectorizing its KNN setup yields zero score delta because the overlay converges so fast that these costs are already negligible.

## How it was tried
- c0007 (cvrp-s11-66566581): restructured overlay loop to use proper DLB and a flat `(route_idx, position, L)` active queue to prevent full route re-scans. Train -0.044, val -0.467. Rejected (tied or noise).
- c0020 (cvrp-s11-66566581): vectorized KNN construction using batch `np.argpartition` to free budget for large-n. Train -0.110, val -0.456. Rejected (tied or noise).
- c0012, c0014, c0020 (cvrp-s13-71671014): added a module-level `_KNN_CACHE` dictionary keyed by `dist.ctypes.data` to reuse the KNN array across the 5+ overlay passes. Train -0.097 to -0.138, all rejected as ties or noise.

## Why it failed
- The current overlay costs ~0.01-0.04s. Even cutting KNN build or route re-scans in half saves milliseconds, which disappears into the eval noise band (see performance-analysis/score-noise-and-gate.md).
- The overlay converges in 1-2 passes, meaning redundant scans are already exceptionally rare.

## Verdict
exhausted. Do not propose Python overlay DLB restructuring, KNN caching, or KNN vectorization again.
