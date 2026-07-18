# Pitfalls

## Contract violations (instant error, score -inf)
- Return a permutation of range(n): every node exactly once, ints (np.int64 is
  fine — the harness casts with int()). Duplicated or missing nodes -> error.
- Do NOT append the start node at the end ([0, ..., 0] is length n+1 -> error).
  The tour is closed implicitly; length includes d[tour[-1], tour[0]].
- Any rotation/direction is accepted — never waste effort canonicalizing.

## Determinism
Runs must be reproducible: use np.random.default_rng(<fixed int>), never
np.random.seed-free global randomness or time-based seeds. Nondeterminism makes
selection noisy and your candidate may be rejected on re-evaluation.

## Complexity budget
- n up to 200 (hidden split up to 150), several instances per split, 60-90 s per
  split, and the harness's own polish + NN baseline run inside the same timeout.
- O(n^2) Python loops are fine; O(n^3) Python loops (e.g. NN from all n starts at
  n=200, or cheapest insertion recomputed naively) will time out — use incremental
  updates or vectorize with numpy.
- np.argsort on the full flattened matrix (greedy edge) is fine: 200*200 = 40k.

## Distance-matrix gotchas
- The diagonal is 0: when taking argmin of a distance row, mask the diagonal and
  visited nodes (e.g. set them to np.inf in a COPY) or you will "visit" yourself.
- Hidden split distances are integers (TSPLIB nint) — many exact ties. Break ties
  deterministically (np.argmin takes the first); don't rely on strict inequality.
- You get a matrix, not coordinates. Geometry tricks (convex hull, k-d trees,
  space-filling curves) are unavailable unless you embed the matrix yourself —
  matrix-only methods (insertion, savings, greedy edge) transfer better anyway.

## Scoring misconceptions
- The reference NN baseline is recomputed in-harness per instance; you cannot
  degrade the baseline, only improve your own tour.
- Improvements must survive the fixed 2-opt polish: shaving a raw tour from 22% to
  20% gap often changes the polished result by ~0 — target structural gains
  (fewer long edges, no isolated misplaced nodes), not micro-optimizations.
