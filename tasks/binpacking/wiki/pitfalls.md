# Pitfalls

## Contract violations (instant -inf)
- **Return shape**: must return one score per entry of `bins` (a 1-D array,
  `len(bins)`). Returning a scalar, a Python list of wrong length, or a 2-D
  array breaks `argmax` placement. `return -(bins - item)` shape is the model.
- **NaN/inf scores**: `np.argmax` on NaNs is garbage-in-garbage-out. Typical
  sources: `1/gap` when `gap == 0` (perfect fit!), `log(gap)` at 0, `exp` of
  large positives overflowing to inf. Use `1/(gap + eps)`, `np.log1p`, clip
  exponents. A perfect fit (`gap == 0`) is *common* and *desirable* — never
  divide by raw `gap`.
- **Name/signature**: the function must be exactly `priority(item, bins)` at
  module top level. Keep the `import numpy as np` line.

## Feasibility misconceptions
- `bins` already contains **only feasible bins** (`bins - item >= 0` guaranteed).
  Do not re-filter, mask, or return `-inf` for "infeasible" bins — there are
  none, and masking can produce all `-inf` rows.
- `bins` is remaining capacity, **not** current load. Load = `100 - bins`.
- Every not-yet-used bin appears with remaining exactly `100.0`; there are many
  of them (one per item, pre-allocated). Ties among empty bins are harmless
  (argmax takes the first), but a heuristic that scores empty bins *highest*
  degenerates to one-item-per-bin — always make an empty bin lose to any
  reasonable partial bin unless nothing fits well.

## Implicit new-bin mistakes
- Best-fit already never opens a bin when a partial bin fits at least as tightly
  (empty bin has the largest gap). If you add bonuses keyed on `bins` (not
  `gap`), check you didn't accidentally make `bins == 100` attractive.
- Off-by-one on thresholds: items are integer-valued floats in [1, 100], gaps
  are integer-valued floats. Test `gap < 0.5` rather than `gap == 0` if you
  fear float dust (here subtraction is exact, both work).

## Performance
- `priority` runs once per item: 5000 calls x 5 instances on `public`. Any
  Python-level loop over `bins` (up to 5000 entries) risks the 120 s timeout.
  Stick to whole-array numpy expressions; avoid `np.vectorize`, list
  comprehensions, and per-bin `if`.
- Do not mutate `bins` in place — it is a view into the packer's state.
- Avoid randomness: it adds evaluation noise across splits and re-evaluations
  for zero expected gain.

## Overfitting
- Constants tuned to items~40 (mean of Weibull(3)*45) — e.g. "penalize gap in
  [25, 45]" — can invert on the private OOD splits (mean ~27 and uniform 20..70).
  Prefer expressing thresholds relative to `item` or capacity 100, and keep the
  best-fit backbone `-gap` as the dominant term.
