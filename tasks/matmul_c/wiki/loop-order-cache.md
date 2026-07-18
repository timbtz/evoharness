# Loop order and cache behavior

Naive i-j-k walks B by column: stride-n doubles, one cache miss per multiply at
n>=192 (row of B = 1.5-4 KB, whole B = 288 KB-2 MB vs 32 KB L1 / 512 KB L2).
That is why the reference kernel runs at only ~1-2 GFLOPS.

**i-k-j reorder** — the single biggest win (typically 2-12x here):

    for (i) {
        for (j) C[i*n+j] = 0.0;
        for (k) {
            double a = A[i*n+k];
            for (j) C[i*n+j] += a * B[k*n+j];   // unit stride in B and C
        }
    }

Inner loop is a saxpy over contiguous B/C rows: gcc -O3 -march=native
auto-vectorizes it to AVX2 FMAs. Measured on this machine: ~11-13x vs naive.

Other orders: j-k-i is as bad as i-j-k (column walks); k-i-j is fine but forces
the C-zeroing pass anyway. Transposing B first (n*n copy, then dot products over
two unit-stride rows) also works: transpose costs O(n^2), negligible vs O(n^3).

Rules of thumb:
- Make the INNERMOST loop unit-stride over the arrays touched most.
- Hoist loop-invariant loads (`double a = A[i*n+k]`) — helps gcc prove no alias.
- `const double* restrict` on A/B and `double* restrict` on C lets gcc vectorize
  without runtime alias checks (signature change is allowed as long as the name
  and argument order stay; restrict is part of the pointer type, callers unaffected).
