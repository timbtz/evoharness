# Multi-Seed Bank Triage
Evaluating truncated/escaped variants of the top-2 distinct bank seeds in a single batched Phase 0 provides structurally distinct escaped basins, yielding a slight improvement in train score without breaking val. However, multi-seed tournaments mixing escaped and unescaped boundaries must strictly separate their `safe_b` tracking, or they will contaminate the polish loop.

## How it was tried
- `stellar_p2-s101-20239089` c0006r1, c0006f: Sorted `fm.seed_bank_info()` by official score to find the top-2 seeds. For seed #0, generated truncation/pivot escapes (keep=(6,6)). For seed #1, applied aggressive truncation (keep=(5,4)) for cheap evals and structural escape. Evaluated up to 6 candidates in a single `fm.eval_many` batch and fed the winner into the proven polish loop.
- `stellar_p2-s102-48117936` c0006 (train -0.247, val -0.453): Ran a 12-probe tournament across bank seeds 3, 0, and 1. Selected the best escape by raw `fm.score`. This proved fatal: an alternative seed's escaped boundary naturally has a higher raw `fm.score` than the incumbent (because it avoids the 0.05 novelty penalty without actually breaking physics depending on hidden walls). When fed into the polish loop, the generic `feasibility <= 0.009` gate selected this base, drifting into deep infeasibility and slipping the LF gate.
- `stellar_p2-s103-71917443` c0002f: Proposed triaging the escape BASE by measured QI margin (deepest log10_qi wins) so out-of-ball dilation/truncation escapes start with head-room below the QI wall. It was rejected for a pure budget-reserve fix that became c0003.

## Why it worked / failed
Batching distinct seeds gives the optimizer empirical data on which escape basin offers a better starting point for polish. However, using raw `fm.score` to select the base across different seeds is strictly dominated by the novelty penalty: the unpenalized alternative seed will look artificially high-scoring and poison the acceptance key. 

## Verdict
promising — Succeeded in pushing train to 0.6161 (from 0.6140) while maintaining val 0.6280. CAVEAT: the deterministic winner froze the whole branch on one boundary (basin monoculture). Portfolio breadth must explicitly track margin-positive escapes, and cross-seed tournaments MUST evaluate bases using the novelty-aware key (`score - 0.05 * max(0, 1 - dist/1e-3)`), never raw score.
