# Worst-Member Interpolation in Phase 1
Pushing the *worst-scored* pool members toward the best elite via linear interpolation to sample Fourier-mode directions.

## How it was tried
- stellar_p2-s11-38566380 c0026 (rejected, -0.5413): Added `_interpolate(b0, b1, alpha, sigma, rng, boost=1.0)` that blends the target `pool[0]` with the worst `pool[li]`. Replaced the uniform elite sampling in Phase 1 with this directed scheme.

## Why it failed
The writer predicted this would sample "geometrically distinct regions" and enter new basins. However, linearly blending a good boundary with a terrible one does not yield a physically intermediate stellarator; it averages their constraint penalties, pulling the geometry out of the smooth, NAE-stable subspace that CMA-ES/gradient descent relies on. It scored -0.5413, falling into the same failure regime as early recombination and mean-crossover attempts (which also blend across distinct basins).

## Verdict
refuted — Do not blend or interpolate across distinct pool members. Stick to zero-RNG geometric transforms (like shear/dilation) or single-elite isotropic Gaussian mutations.
