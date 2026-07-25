# Guidance for refiner sessions (process evidence, keep this page)
Plain reverts to the incumbent are near-wasted slots: they tie the incumbent
(-0.4394 basin) and add no information.

## Evidence (stellar_p2-s11-38566380)
- c0002f/c0003f/c0010f/c0013f/c0019f/c0020f: revert-style refinements — every
  one landed exactly on the incumbent endpoint or the -0.4412/-0.4395
  attractors. Zero information gained about new basins.
- c0004f (sign-alternating probes, -0.4412) and c0011f (direct attack on the
  binding constraint, -0.505) were ORIGINAL mechanisms — rejected but they
  mapped new attractor points, which reverts never do.

## Evidence (stellar_p2-s102-48117936, refiner claude-opus-4-8)
- 16 refinements: 7 revert-to-incumbent ties (bit-exact 0.5789), 5 val-negative,
  none beat the branch best. The one new-best (c0003f, +0.0145 val over the seed)
  was a revert PLUS three surgical wiki-grounded additions — reverts only pay
  when they carry a distinct mechanism.
- c0013f made the run's sharpest diagnosis (in-ball mirage: 0.6057 − 0.0268
  penalty = the 0.5789 "tie") from metrics alone — but aimed the fix at the
  return gate when the binding failure was escape supply. Diagnose WHERE the
  failure lives (generator vs gate) before spending the slot.

## What a refiner should do
When the writer's idea is broken: REPAIR its mechanism so the idea genuinely
executes. When it is refuted: propose YOUR OWN distinct improvement from the
index's Open directions (early-trajectory phases decide the outcome here).
Never submit a near-copy of the incumbent.
