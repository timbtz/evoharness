# Optimization: pure-numpy iterative improvement

No scipy. Everything below is numpy-only and fits easily in the 60 s budget
for n=26 (a 26x26 distance matrix is microscopic; ~50k light steps are fine).

## Core loop: relax centers, re-solve radii
```python
rng = np.random.default_rng(0)            # determinism is REQUIRED
best_c, best_r = c.copy(), max_radii(c)
for it in range(2000):
    r = max_radii(c, iters=30)
    # repulsion: push apart pairs that are "tight" (r_i + r_j ~ d_ij)
    diff = c[:, None] - c[None]           # (n, n, 2)
    d = np.sqrt((diff ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    tight = (r[:, None] + r[None]) / d    # ~1 where binding
    push = (diff / d[..., None] * np.maximum(tight - 0.98, 0)[..., None]).sum(1)
    c = np.clip(c + 0.02 * push, 0.0, 1.0)
    if r.sum() > best_r.sum():
        best_c, best_r = c.copy(), r.copy()
```
Ideas that matter more than the exact formulas:
- **Always re-solve radii after moving centers** — score lives in the radii.
- **Track the best** configuration seen; return `best_*`, not the last iterate.
- **Step size decay** (`0.02 * 0.999**it`) stabilizes late convergence.

## Escaping local optima
- **Perturb-and-reinflate**: every k iterations, jitter centers with seeded
  noise `c += rng.normal(0, sigma, c.shape)` (sigma ~ 0.01–0.03), re-solve
  radii, keep only if the best improves (or anneal sigma down).
- **Teleport the worst**: move the circle with smallest radius to the largest
  empty gap (e.g. the point maximizing distance to all circles/walls on a
  coarse grid probe) — reallocates wasted circles.
- **Multi-start**: try 3–5 parametrized layouts (row patterns 5-6-5-6-4,
  6-5-6-5-4, grid+corners), optimize each briefly, refine the winner. Budget:
  measure with a fixed iteration count, not wall-clock (determinism).

## Coordinate/pattern search (derivative-free, very robust)
For each circle, try moving its center by ±h in x and y (8 candidates),
re-solve radii locally, keep the best move; shrink h when no move helps
(h: 0.02 → 1e-4). Slower per step but monotone — good as a final polish.

## Projection and feasibility maintenance
- After every center update: `c = np.clip(c, 0, 1)` (radii handle walls).
- Guard coincident centers before dividing: `d = np.maximum(d, 1e-12)`.
- Final return: `r = np.maximum(r - 1e-10, 0)` (or `r *= 1 - 1e-9`) so
  float error never trips the 1e-9 validator tolerance.

## Budgeting the 60 s
Vectorized inner loop cost is dominated by `max_radii` sweeps. Rough guide:
2k outer iterations x 30 Gauss-Seidel sweeps runs in seconds. If you add
nested Python loops per pair (26*25/2 = 325 pairs) keep total iterations
under ~1e7 primitive ops. Never busy-wait on time; count iterations.
