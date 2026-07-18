# Blocking, unrolling, SIMD

**Cache blocking (tiling).** Above ~n=256, B no longer fits in L2 (512 KB on
EPYC-Rome) and i-k-j starts re-streaming it n times. Block k and j so the
working set (one A panel + one B block) stays in L1/L2:

    for (kk = 0; kk < n; kk += KB)
        for (jj = 0; jj < n; jj += JB)
            for (i) for (k in kk..) { a = A[i*n+k]; for (j in jj..) C += a*B; }

Good sizes here: KB in {64,128}, JB in {64,128,256} (n is a multiple of 32, so
no edge cases). Expect +20-60% over plain i-k-j at n=384/512; little effect at 128.

**Register blocking / unrolling.** Compute a small C tile (e.g. 2 rows x 8 cols)
entirely in local variables inside the k loop: halves loads of B, hides FMA
latency (4-5 cycles, 2 FMA ports -> need ~10 independent accumulators).
Unroll k by 4 with separate partial sums when accumulating a scalar dot product.

**SIMD.** -march=native gives AVX2+FMA (EPYC-Rome, 256-bit: 4 doubles/op,
peak ~16 flops/cycle/core). gcc auto-vectorizes clean unit-stride loops —
check you didn't break that before hand-writing intrinsics. If explicit:
`#include <immintrin.h>`, `_mm256_loadu_pd / _mm256_fmadd_pd / _mm256_storeu_pd`
on the j loop, 2-4 accumulators. No aligned-load tricks needed (loadu is cheap).

Ceiling: single-core peak ~35-40 GFLOPS; OpenBLAS reaches ~30+. A blocked,
register-tiled i-k-j typically lands 18-25 GFLOPS (12-20x naive). Diminishing
returns are real — prefer simple structures the compiler vectorizes over
fragile intrinsic walls.
