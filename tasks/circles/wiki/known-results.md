# Known results for n=26 (unit square, maximize sum of radii)

## Targets
- **Best known: sum ≈ 2.635** — AlphaEvolve (DeepMind, 2025).
- **OpenEvolve reproduction: 2.634292** (ratio 0.99973 of target) — reached
  via evolved constructions + an SLSQP polish (scipy is NOT available here;
  the same quality is approachable with the pure-numpy loops in
  optimization.md, and ≈2.63 was already reached with constructed layouts
  plus max-radii assignment before the scipy step).
- The task reports `metrics.ratio = sum_radii / 2.635`; anything ≥ 0.99 is a
  strong result, ≥ 2.60 is very good, ≥ 2.635 would match the state of the art.

## Progression ladder (OpenEvolve's actual milestones — useful waypoints)
| approach                                   | sum of radii |
|--------------------------------------------|--------------|
| center + 2 concentric rings, naive         | 0.959        |
| tuned hex ring construction                | 1.795        |
| uniform 6x5 grid, uniform r (our seed)     | ~2.16        |
| staggered grid, variable radii, 50-step opt| 2.201        |
| constructed layouts plateau                | ~2.377       |
| + real numerical optimization of centers   | 2.634        |

Lesson encoded in that table: *changing the approach* (construction → radii
LP → center optimization) beats tuning constants at every plateau.

## What a ~2.63 packing looks like
- Mixed radii, roughly 0.02–0.17; mean radius ≈ 2.635/26 ≈ 0.101.
- A few large circles occupy the interior; medium circles line the edges;
  small circles fill the 4 corners and curvilinear gaps between big ones.
- Nearly every circle is "tight": touching (within tolerance) either two+
  neighbors or a wall and a neighbor. Slack anywhere = wasted radius.
  A quick audit: count binding constraints `d_ij - (r_i+r_j) < 1e-6` and
  `wall_i - r_i < 1e-6`; loose circles are optimization targets.
- Contrast with equal-radius packing: 26 equal circles give r ≈ 0.084 each,
  sum ≈ 2.19 — the extra ~0.45 comes entirely from *unequal* radii.

## Sanity anchors
- Seed (6x5 grid, r = 0.5/6 - 1e-4): sum ≈ 2.164. Any submission below this
  is a regression; the evaluator treats invalid output as -inf, so a valid
  2.2 always beats an invalid 2.64.
- Upper bound intuition: total circle area <= 1 gives no tight bound for sum
  of radii (sum r is maximized by many *unequal* circles), so use 2.635 as
  the practical ceiling rather than area arguments.

Attribution: milestone numbers and layout observations adapted from
openevolve/examples/circle_packing (Apache-2.0); best-known value from the
AlphaEvolve paper as cited there.
