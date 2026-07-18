# Heuristic families for online bin packing priorities

All heuristics below are expressed as `priority(item, bins) -> np.ndarray` where
`bins` holds remaining capacities of feasible bins only (`bins - item >= 0`).
Define `gap = bins - item` (always `>= 0`).

## Classical baselines
- **Best-fit**: `return -gap` — pick the tightest bin. ~4% excess on this task. The baseline to beat.
- **First-fit-ish**: favor the fullest (lowest remaining) bin regardless of fit: `return -bins`. On this task it behaves close to best-fit, usually slightly worse.
- **Worst-fit**: `return gap` — spreads items out; bad for excess (leaves many half-full bins). Useful only as a negative example.
- **Harmonic/interval**: classify item into size classes (e.g. (50,100], (33,50], (25,33]...) and reserve bins per class. Hard to express in a single stateless priority; borrow the *idea*: score bins whose load class "matches" the item class higher.

## FunSearch-style evolved ideas (these reach <1% excess)
1. **Near-perfect-fit bonus** — a large reward when the item (almost) fills the bin:
   `score = -gap; score[gap < 2] += 100.0` or smooth: `score = -gap + 50*np.exp(-gap)`.
2. **Awkward-residual penalty** — penalize leaving mid-size gaps (~15–45 here) that no
   future item pairs well with: `score -= 8*np.exp(-((gap-30)**2)/150)`.
   Small residuals (0–5, wasted anyway) and large residuals (still usable) are fine.
3. **Nonlinear gap transforms** — sharpen best-fit's preference:
   `-(gap)**2`, `-np.sqrt(gap)`, `1.0/(gap + 0.1)`, `-np.log1p(gap)`.
   Convex transforms (`gap**2`) punish loose fits harder than linear best-fit.
4. **Discourage opening new bins** — empty bins have `bins == 100`; subtract a
   constant or scaled penalty: `score -= np.where(bins == 100.0, 15.0, 0.0)`.
   Every empty bin is a candidate, so this single term controls bin opening.
5. **Relative/scale-free features** — `gap/item`, `gap/100`, `(bins - item)/bins`.
   These generalize better to other item distributions than absolute thresholds.

## Composition template (a strong known shape)
```python
def priority(item, bins):
    gap = bins - item
    score = -gap                       # best-fit backbone
    score += 40.0 * (gap < 1.0)        # perfect/near-perfect fit bonus
    score -= 10.0 * np.exp(-((gap - 28.0) ** 2) / 200.0)  # awkward residual
    score -= 12.0 * (bins == 100.0)    # delay opening fresh bins
    return score
```
Tune magnitudes so the tiers don't drown each other: fit bonus >> best-fit slope
over its range (100) >> residual penalty >> new-bin penalty is a common ordering
mistake — in practice the *new-bin penalty and fit bonus* matter most.

## Search directions that historically pay off
- Piecewise scores keyed on gap thresholds (0, 1, 2, item-relative cuts).
- Multiplying rather than adding terms: `-(gap+1) * (1 + 0.5*(bins==100))`.
- Using `item` to modulate: big items (>=50) should strongly prefer tight fits;
  small items can afford to top up nearly-full bins (`bins < 20`).
- Keep it stateless and vectorized; no loops, no globals mutated across calls.
