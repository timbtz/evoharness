# The Shallow Uncontraction Trap (Honest-Margin Ladders)
Sweeping shallower depths (reducing `b1`/`b2` magnitude) to trade raw score for a lower feasibility margin strictly fails on the honest-score metric.

## How it was tried
- `stellar_p2-s100-78100567` c0022 (REJ, train 0.5582, val 0.5647): Attempted a partial-uncontraction ladder targeting honest_score > incumbent. The selected candidate scored a catastrophic 0.558 train.
- `stellar_p2-s100-78100567` c0025 (REJ, train 0.5582, val 0.5647): Re-attempted the shallow-contraction ladder with 20-35% reduced depth. Scored identically to c0022, crashing to the floor.

## Why it failed
The writer predicted that reducing depth would reduce aspect ratio, saving ~0.92 score/feas and landing at a Pareto-superior honest_score. The code evaluated exactly this. However, because `feasibility` is dominated by the aspect ratio constraint, uncontracting forces the boundary away from the aspect wall. This abruptly collapses the baseline `objective_L` (and thus `p2_score`) significantly faster than the linear 0.92 exchange rate recovers feasibility margin.

## Verdict
refuted — Stop trying to buy an "honest margin" by reducing the contraction depth. The physics strictly rewards sitting as deep into the aspect ratio tolerance wall as possible (feas ~0.009+).
