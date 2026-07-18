# Circle packing wiki (n=26, unit square, maximize sum of radii)

Pages:

- `constructions.md` — initial layouts (grid, hex, rings) and the greedy
  max-radii assignment that turns any center set into a valid packing.
- `optimization.md` — pure-numpy iterative improvement: repulsion/relaxation,
  perturbation escapes, projection/clipping, radius re-inflation loops.
- `known-results.md` — best known sums for n=26, what good packings look like.
- `pitfalls.md` — common ways candidates fail validation or time out.

Contract reminder: `pack(n)` returns an `(n, 3)` array of `(x, y, r)`;
containment and pairwise non-overlap are checked with tolerance 1e-9;
score = sum of radii; 60 s time limit; must be deterministic.
