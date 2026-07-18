# Pitfalls

- **C is not zeroed on entry.** Any accumulate-style kernel (i-k-j, blocked)
  must write C[i*n+j] = 0.0 first (or assign on first k-block). Reading stale C
  = wrong result at the first check.
- **Do not define main(), naive(), rnd(), now(), or bench()** — the harness owns
  them; redefinition is a duplicate-symbol compile error, scored -inf.
- **Tolerance is 1e-7*n absolute.** Reassociated sums (blocking, SIMD, multiple
  accumulators) differ from naive by ~1e-13 — fine. float instead of double, or
  skipping the tail of a loop, is not.
- **Keep the signature** `void matmul(const double*, const double*, double*, int)`
  (adding `restrict` is fine). n is a multiple of 32 in [128,512] — but do not
  hardcode one n; the private split uses sizes you were not timed on.
- **Aliasing stalls vectorization.** Without `restrict`, gcc may emit runtime
  overlap checks or scalar fallbacks. Hoisting A[i*n+k] into a local also helps.
- **Timing is same-binary relative** (best-of-2 batches, auto reps): machine
  noise mostly cancels, but sub-5% "gains" are likely noise — aim for
  structural wins (order, blocking, tiling), not micro-jitter.
- **-O3 -march=native, single thread.** `#pragma omp` is ignored (no -fopenmp);
  threads/forks are pointless (1 CPU rlimit) and may hit the memory cap.
  gcc already unrolls, vectorizes clean loops, and fuses FMAs: measure before
  adding complexity — a simpler kernel that vectorizes beats clever code that
  does not.
- **No I/O, no globals with constructors.** Print nothing; the harness prints
  the one JSON line.
