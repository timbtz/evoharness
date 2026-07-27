# Radial Translation, NAE Basins, and Iterative Coordinate Descent
Attempting to find orthogonal escape mechanisms—such as shifting the boundary's centroid, evaluating structurally distinct NAE basins, or using adaptive coordinate descent—fails to beat the two-stage contraction winner and often risks budget starvation.

## How it was tried
- `stellar_p2-s105-26196944` c0026 (REJ, train 0.5582): Attempted an adaptive three-stage coordinate-descent over `(cr, b2, c2)` using sequential `eval_many` calls instead of a static grid. Starved the budget and dropped to the floor.
- `stellar_p2-s105-26196944` c0028 (REJ, train 0.6255): Added an independent nfp=3 NAE-seeded escape boundary (`fm.seed_nae`) alongside B3 and applied the proven two-stage contraction. Regressed to the floor.
- `stellar_p2-s105-26196944` c0032 (REJ, train 0.6255): Applied an m-differential radial translation (`r_cos[m][center] += shift * sqrt(m/mpol)`) to shift the boundary's centroid outward at high-m. Regressed.

## Why it failed
Writers predicted that exploring new search mechanics (coordinate descent), new structural basins (NAE), or orthogonal geometric shifts (radial translation of the n=0 column) would uncover Pareto-superior aspect/QI tradeoffs. The code executed these exactly, but:
1. Iterative loops structurally betray the batched `eval_many` design needed for budget safety (budget-discipline.md).
2. The NAE basin simply lacks the baseline objective_L of the B3-lhhhhappy3 escape (surrogate-and-nae-escapes.md).
3. Radial translation disrupts the delicate major-radius-to-aspect coupling that smooth contraction preserves.

## Verdict
exhausted — Stick to the single batched grid on the B3 nfp=3 basin. Stop trying iterative descent, NAE basins, or major-radius translations.
