# Scoring: L1 bound, excess metric, reference numbers

## Metric
For each instance: `used` = number of bins actually opened,
`L1 = ceil(sum(items) / 100)` (perfect-fill lower bound, capacity 100).

- `excess = (used - L1) / L1` per instance
- `score = -mean(excess)` over the split's instances (higher is better, 0 = optimal)
- `metrics.excess_pct = 100 * mean(excess)`

L1 is not always achievable (fractional waste is unavoidable), so excess ~0.1–0.5%
is effectively perfect for this distribution; exactly 0 is not a realistic target.

## Reference numbers (public split, Weibull(3)*45, 5000 items x 5 instances)
| heuristic                              | score   | excess_pct |
|----------------------------------------|---------|------------|
| worst-fit `gap`                        | ~ -0.10+| >10        |
| first-fit-ish `-bins`                  | ~ -0.05 | ~5         |
| best-fit `-(bins-item)` (the seed)     | -0.0399 | 3.99 (4.59 on train) |
| evolved (fit bonus + residual penalty) | > -0.01 | < 1        |

Scale intuition: mean item ≈ 40.2, so 5000 items sum ≈ 201k → L1 ≈ 2010 bins.
1% excess ≈ 20 extra bins per instance; each avoided bin ≈ 0.05 excess_pct.

## Splits and what is measured where
- `train`: 5 instances, 1000 items, seeds 2024–2028 — fast feedback (~seconds).
- `val`: same distribution, seeds 3024–3028 — checks you didn't overfit seeds.
- `public`: 5 instances, 5000 items, seeds 4024–4028 — the headline number.
- `private`: out-of-distribution (Weibull(1.5)*30 and uniform 20..70) — measures
  generalization. Absolute-threshold heuristics tuned to items~40 can regress
  badly here; scale-free terms (`gap/item`, `gap/bins`) hold up better.

## Practical scoring notes
- Score differences of <0.001 (0.1 excess_pct) between candidates on `train`
  (only 1000 items, L1 ≈ 402) are noise-level; confirm on `val` or `public`.
- Timeouts: 60 s for 1000-item splits, 120 s for 5000-item splits. The packing
  loop calls `priority` once per item; keep the function to a handful of
  vectorized numpy ops or you will time out before scoring at all.
- An error (exception, timeout, malformed output) scores `-inf` — a working
  mediocre heuristic always beats a clever broken one.
