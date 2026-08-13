# g3 run: Bank Seed Retrieval Traps and Aspect-Margin Games
In run `g3`, attempts to retrieve authentic bank seeds or play aspect-margin games via `fm.margin_step` consistently failed to recover the authentic baseline `objective_L`, capping the run at the isolated state floor.

## How it was tried
- `g3` c0001 (REJ, train 0.5697): Replaced the hardcoded B3 matrix with dynamic retrieval of `fm.seed_bank(3)`. Regressed because the bank seed retrieval failed or was uncompetitive in this isolated harness state.
- `g3` c0003 (ACC, train 0.5843): Prepended `seed_bank(3)` to the portfolio while keeping the hardcoded B3 fallback. Safely tied the parent floor as the bank seed was inferior or inaccessible.
- `g3` c0006 (REJ, train 0.5733): Replaced the portfolio with a single densely-sampled multi-seed R/Z-split contraction sweep. Regressed slightly.
- `g3` c0010 (REJ, train 0.5805): Restored the exact B3 matrix and added opportunistic bank seed prepends. Regressed slightly.
- `g3` c0012 (ACC, train 0.5843): Restored the exact parent pipeline but hardened the acceptance key to strictly guard against None/soft-fail metric corruption. Tied the parent floor.
- `g3` c0013 (REJ, train 0.5646): Added an L-augmenting, margin-preserving pipeline using `fm.metric_grad` to intelligently shape spectral coefficients. Regressed, confirming gradients fail on isolated boundaries.

## Why it failed
Writers assumed that retrieving the authentic bank seed at runtime would seamlessly replace the fabricated matrix. The code applied this directly. However, in this isolated harness state, the dynamic retrieval either failed to provide the authentic objective_L or disrupted the proven fallback chain. Attempting to optimize via `metric_grad` or denser grids on the available anchors simply bounced off the structural physics cap.

## Verdict
exhausted — Stop relying on dynamic bank seed retrieval to save isolated runs. The binding constraint remains the exact hardcoded elite matrix.
