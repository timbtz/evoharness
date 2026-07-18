# Matmul C kernel wiki (double, row-major, single core, gcc -O3 -march=native)

- Loop order & cache — why i-k-j beats i-j-k, streaming access, ~2-12x — loop-order-cache.md
- Blocking, unrolling, SIMD — tiling for L1/L2, register blocking, FMA/AVX2 — blocking-simd.md
- Pitfalls — correctness tolerance, C not zeroed, aliasing, what gcc already does — pitfalls.md

Contract: `void matmul(const double* A, const double* B, double* C, int n)`,
n multiple of 32 in [128, 512]; score = geometric-mean measured speedup vs the
naive i-j-k kernel compiled in the same binary; correctness vs naive, tol 1e-7*n.
