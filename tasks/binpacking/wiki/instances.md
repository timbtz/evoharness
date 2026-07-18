# Instances: item distribution and generalization

## Generator (exact)
```python
rng = np.random.default_rng(seed)
x = rng.integers(20, 71, n) if shape == 0 else rng.weibull(shape, n) * scale
items = np.clip(np.round(x), 1, 100).astype(float)   # integer-valued floats
```
Capacity is always 100. Placements are online and final; a fresh bin is always
among the feasible candidates.

## Train/val/public distribution: Weibull(shape=3, scale=45)
- mean ≈ 40.2, std ≈ 14.6, mode ≈ 39 — a mildly left-skewed bell.
- Bulk of mass in [15, 65]; items > 80 or < 10 are rare (<1%).
- Typical bin holds 2–3 items (2 x ~40 leaves gap ~20; 40+40+20 fills).
  The game is mostly: pair mid items tightly, use small items (10–25) to top
  up residuals ~10–25, and avoid stranding residuals ~25–45 that only rare
  items can complement.
- Instance sizes: train/val 1000 items (L1 ≈ 402), public 5000 (L1 ≈ 2010).
  Seeds: train 2024–2028, val 3024–3028, public 4024–4028.

## Private split is OOD — two unseen distributions
1. **Weibull(shape=1.5, scale=30)**, seeds 5024–5025: mean ≈ 27, std ≈ 18,
   strong right skew — many small items (mode ≈ 15), occasional 60–100 items.
   Bins hold 3–4+ items; small-item top-up matters much more, and "awkward
   residual" is a different range than for the train distribution.
2. **Uniform integers 20..70**, seeds 5026–5027: mean 45, flat. Frequent exact
   complements (30+70, 45+55); tight two-item fits dominate.

## What generalizes (and what does not)
- Generalizes: best-fit backbone `-gap`; near-perfect-fit bonuses keyed on
  small absolute `gap` (0–2); penalizing gaps *relative to the current item*
  or to capacity (e.g. `gap/item`, `gap/100`); new-bin penalty via
  `bins == 100`.
- Fragile: hard-coded windows like "penalize gap in [25,45]" (tuned to
  mean-40 items), assumptions that every bin ends with 2 items, bonuses for
  residuals equal to specific magic numbers near 40.
- Aim for a heuristic whose *shape* is distribution-free with a few mild
  constants; validate that train/val scores agree before trusting a constant.

## Handy facts
- `items.sum()/100` per instance: ~402 (train, n=1000), ~2010 (public),
  ~1355 (Weibull 1.5, n=5000), ~2250 (uniform, n=5000).
- All items are integers in [1, 100] after rounding/clipping, so gaps are
  integer-valued; `gap < 0.5` == "perfect fit".
- Everything is seeded and deterministic — identical code always gets the
  identical score on a given split.
