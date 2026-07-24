# Python overlay DLB fixes, KNN caching, and KNN vectorization (micro-optimizations)

Refactoring the Python Or-opt overlay's don't-look-bits (DLB), caching KNN arrays, vectorizing its KNN setup, consolidating loop bookkeeping, or replacing the two-tier overlay with a single unified pass yields zero score delta because the overlay converges so fast that these costs are already negligible.

## How it was tried
- c0007 (cvrp-s11-66566581): restructured overlay loop to use proper DLB and a flat `(route_idx, position, L)` active queue to prevent full route re-scans. Train -0.044, val -0.467. Rejected (tied or noise).
- c0020 (cvrp-s11-66566581): vectorized KNN construction using batch `np.argpartition` to free budget for large-n. Train -0.110, val -0.456. Rejected (tied or noise).
- c0012, c0014, c0020 (cvrp-s13-71671014): added a module-level `_KNN_CACHE` dictionary keyed by `dist.ctypes.data` to reuse the KNN array across the 5+ overlay passes. Train -0.097 to -0.138, all rejected as ties or noise.
- c0013 (run cvrp-s29-99479842): Replaced `_py_fast_oropt` with `_unified_overlay` merging Or-opt + 2-opt + swap into a single loop. Train -0.082, val -0.414.
- c0015 (run cvrp-s29-99479842): Merged `_py_fast_oropt`, `_py_or_opt1`, `_py_segment_shift` into one tight budget-fill `_py_unified_overlay`. Train -0.139, val -0.238.
- c0019 (run cvrp-s29-99479842): Replaced fragmented two-tier overlay with `_py_unified_oropt_2opt_swap` sharing bookkeeping to save ~0.3-0.5s overhead. Train -0.0838, val -0.2526.
- c0020 (run cvrp-s31-5170619): Deleted `_py_fast_oropt` (Tier 1) to replace two-tier system with a single unified budget-gated pass. Train -0.151, val -0.262. Refuted: combining tiers does not unlock extra search time for the C kernel, as the C LNS already saturates its budget limit.

## Why it failed
- The current overlay costs ~0.01-0.04s. Even cutting KNN build or route re-scans in half saves milliseconds, which disappears into the eval noise band (see performance-analysis/score-noise-and-gate.md).
- The overlay converges in 1-2 passes, meaning redundant scans are already exceptionally rare.
- Unified loops and shared bookkeeping do not unlock any extra search time for the C kernel, as the C LNS already saturates its budget limit.

## Verdict
exhausted. Do not propose Python overlay DLB restructuring, KNN caching, KNN vectorization, structural consolidations, or unified single-pass rewrites again.
