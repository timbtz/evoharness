# Heuristic families for online bin packing priorities

All heuristics below are expressed as `priority(item, bins) -> np.ndarray` where
`bins` holds remaining capacities of feasible bins only (`bins - item >= 0`).
Define `gap = bins - item` (always `>= 0`).

## When does a term actually change anything? (read this first)
Only the ORDERING of scores matters — the item goes to the argmax bin. Two facts
kill most "improvements" silently (the candidate ties its parent bit-for-bit):
- Any strictly monotone transform of a heuristic IS that heuristic: `-gap`,
  `-gap**2`, `1/(gap+0.1)`, `-np.log1p(gap)` pick the identical bin every time.
  Nonlinear transforms only matter inside SUMS with other terms, where curvature
  changes the trade-off.
- Gaps are integers (items are rounded), so the `-gap` backbone separates
  adjacent candidates by >= 1. An added term flips a decision only where its
  slope exceeds ~1 per unit of gap (or it jumps discretely by > 1). Example:
  `-8*np.exp(-((gap-30)**2)/200)` has max slope ~0.5 — added to `-gap` it never
  changes a single placement. Likewise `+40*(gap < 1)` rewards the bin best-fit
  already ranks first, and `-12*(bins == 100)` demotes the bin `-gap` already
  ranks last: alone, each is dead weight. Make terms steep enough to reorder, or
  combine them so they interact.

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
3. **Nonlinear gap transforms** — useful ONLY summed with other terms (alone they
   are order-identical to best-fit, see above): `-(gap)**2 + bonus`,
   `-np.log1p(gap) - penalty`. Curvature decides how strongly the other terms
   can override the tight-fit preference at different gap sizes.
4. **Discourage opening new bins** — empty bins have `bins == 100`; subtract a
   constant or scaled penalty: `score -= np.where(bins == 100.0, 15.0, 0.0)`.
   Every empty bin is a candidate, so this single term controls bin opening.
5. **Relative/scale-free features** — `gap/item`, `gap/100`, `(bins - item)/bins`.
   These generalize better to other item distributions than absolute thresholds.

## Composition template (a strong known shape)
```python
def priority(item, bins):
    gap = bins - item
    score = -gap                        # best-fit backbone
    score += 50.0 * np.exp(-gap)        # near-fit bonus: slope 50 at gap 0, reorders hard
    score -= 15.0 * np.exp(-((gap - 28.0) ** 2) / 60.0)  # awkward residual, max slope ~1.7
    score -= 12.0 * (bins == 100.0)     # new-bin penalty (interacts with the terms above)
    return score
```
Every term here is steep or large enough to actually flip placements (check the
slope rule above before tuning). In practice the *new-bin penalty and near-fit
bonus* matter most; the residual penalty fine-tunes which non-tight bin wins
once the penalty pushes a mid-gap bin below the empty bin.

## Search directions that historically pay off
- Piecewise scores keyed on gap thresholds (0, 1, 2, item-relative cuts).
- Multiplying rather than adding terms: `-(gap+1) * (1 + 0.5*(bins==100))`.
- Using `item` to modulate: big items (>=50) should strongly prefer tight fits;
  small items can afford to top up nearly-full bins (`bins < 20`).
- Keep it stateless and vectorized; no loops, no globals mutated across calls.
