# Benchmarks and reference numbers

## Scoring recap
gap_pct = 100 * (polished_len - ref) / ref, score = -mean(gap_pct).
Generated splits: ref = polished nearest-neighbor-from-0 (so seed scores ~0, and a
score of +3 means 3% shorter than the polished-NN baseline on average).
Hidden split: ref = the true optimum, so scores there are always <= 0.

## Hidden split: classic TSPLIB (EUC_2D, distances = nint(euclidean))
| instance | n   | optimum |
|----------|-----|---------|
| eil51    | 51  | 426     |
| berlin52 | 52  | 7542    |
| st70     | 70  | 675     |
| kroA100  | 100 | 21282   |
| ch150    | 150 | 6528    |

Polished NN-from-0 gaps on these five: ~2.8, 6.9, 7.6, 9.2, 3.7 % (mean ~6.0%).
Good constructions + Or-opt reach 1-3% mean; matching the optima exactly is not
expected at this budget.

## Rules of thumb (uniform random points, unit square)
- Expected optimal tour length ~ 0.7124 * sqrt(n) (BHH constant): ~5.0 at n=50,
  ~7.1 at n=100, ~10.1 at n=200. Polished NN is ~5-8% above that.
- Raw construction gaps vs optimum (typical): NN 20-25%, greedy edge 15-20%,
  farthest insertion 7-12%, Christofides ~10% (needs matching — too heavy here).
- After the fixed 2-opt polish: NN -> 5-9%, greedy edge -> 3-5%,
  farthest insertion -> 3-6%; +Or-opt and restarts -> 1-3%.

## What a good score looks like
- train/val/public: +2 to +5 (i.e. 2-5% better than the polished NN reference).
- Anything below 0 on generated splits means the construction is worse than NN —
  usually a bug or a degenerate tour order, not an interesting heuristic.
