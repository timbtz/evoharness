# Constructions: initial layouts and max-radii assignment

A packing = centers + radii. Strategy that works: pick a good *center layout*,
then compute the largest valid radii for those centers, then optimize centers.

## Center layouts for n=26 (unit square)
- **Uniform grid 6x5** (the seed): centers `((i+0.5)/6, (j+0.5)/5)`, uniform
  r = 0.5/6 - eps ≈ 0.0832 → sum ≈ 2.16. Simple, valid, far from optimal.
- **Staggered/hex rows**: rows of 5-6-5-6-4 (or 5-6-5-6-5 dropping one), each
  row offset by half a spacing, row pitch ≈ spacing * sqrt(3)/2. Hex packing
  wastes less space than square grid; with per-circle max radii this jumps
  well past the uniform grid (~2.2+ before any center optimization).
- **Center + rings**: one big central circle, ring of 6–8, outer ring of 16
  (+ corners). Historically ~0.96 naive → ~1.8 tuned (OpenEvolve gen 10).
  Inferior to grids for *sum of radii*; rings underuse corners.
- **Size-mixed layouts** (what winners look like): a few large circles
  (r ≈ 0.10–0.17) in the interior, medium ones along edges, small ones wedged
  into the 4 corners and interstitial gaps. Best known sum ≈ 2.635.

## Max radii for fixed centers (LP-free, pure numpy)
For fixed centers, radii maximization is a linear program:
maximize sum r s.t. `r_i <= wall_i` and `r_i + r_j <= d_ij`.
A fast approximation — iterate to a fixed point (order matters, Gauss-Seidel):
```python
def max_radii(c, iters=100):
    n = len(c)
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    wall = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]),
                      np.minimum(c[:, 1], 1 - c[:, 1]))
    r = wall.copy()
    for _ in range(iters):
        for i in range(n):                      # n=26, cheap
            r[i] = min(wall[i], (d[i] - r).min())
    return np.clip(r, 0.0, None)
```
Start from wall distances and shrink; converges in a few sweeps. Variants:
process circles largest-first, or split slack `d_ij` unevenly (e.g. weight by
current r) instead of `d_ij - r_j` to favor large interior circles — equal
split `d/2` per pair is a common suboptimal habit.

## Uniform scale-to-validity (crude but safe)
Given any centers + candidate radii, make it valid by one global scale:
`s = min(1, min(wall/r), min over pairs of d_ij/(r_i + r_j)); r *= s * (1 - 1e-9)`.
Useful as a final safety net before returning.

## Practical recipe
1. Build a staggered layout (parametrize row counts/offsets).
2. `r = max_radii(centers)`.
3. Feed to the optimizer (see optimization.md) to move centers.
4. Shrink radii by ~1e-10 before returning to stay inside tolerance 1e-9.
