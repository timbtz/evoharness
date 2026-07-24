# Free Seed Slot Extremes
Replacing duplicate or inert seed specs in the `SEED_BUDGET=16` portfolio with extreme mirror-crushing geometries at zero budget cost.

## How it was tried
- stellar_p2-s11-38566380 c0026f ACC -0.4392: The incumbent `c0025` featured a duplicate `(5, 2, 8.0)` in slot 13. Because the simulator is deterministic, evaluating this duplicate again was a totally wasted budget unit that could never advance past stable sort tie-breaking. This candidate swapped the duplicate for an extreme `(6, 1, 10.0)` (nfp=6, aspect_ratio=10.0, rt=1.8) seed. 

## Why it worked
Changing the seed spec consumes no RNG during the seeding phase. Thus, if the new `nfp=6` seed failed to outscore the tail of the pool, the entire downstream search trajectory would be byte-identical to the incumbent (guaranteeing a tie at worst). By providing an extreme geometry targeting the mirror ratio, it shifted the early basin just enough to secure a marginal victory (-0.4392 vs -0.4394).

## Verdict
promising — Always sweep the `SEED_BUDGET` array for exact duplicates or inert specs. Use the freed slot to test extreme `nfp` or `aspect_ratio` combinations. (Note: `c0027f` inherited and accepted this, but it was superseded by shear/dilation as the primary constraint-breaking mechanism).
