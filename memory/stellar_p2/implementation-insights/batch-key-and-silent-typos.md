# Batch Evaluation Key Consistency and Silent Typo Traps
Mixing dynamically generated NAE seeds with hardcoded B3 escape matrices in the same `eval_many` batch frequently triggers `KeyError` crashes if their dictionary keys mismatch, while manual code edits introduce tiny float changes that corrupt the exact incumbent matrix.

## How it was tried
- `stellar_p2-s100-78100567` c0005 (ERR): Added NAE seed generation and batch construction. The `eval_many` batch dictionaries from the NAE seeds lacked the `'r_cos'` key expected by the hardcoded matrix evaluation logic, crashing immediately with `KeyError: 'r_cos'`.
- `stellar_p2-s100-78100567` c0005f (ERR): Reverted to pure B3 contraction grids but left a dangling call to the deleted `_dedup_tagged()` helper function. Crashed with `NameError: name '_dedup_tagged' is not defined`.
- `stellar_p2-s100-78100567` c0003, c0004, c0006, c0008, c0009, c0010, c0011i0, c0052-c0061: All introduced silent float typos in the hardcoded `_B3_RCOS` matrix during `diff` generation (e.g., `-...995` mutated to `-...99`, or `-...5503` mutated to `-...5303`). VMEC forgave these tiny errors as valid, but they silently destroyed the exact incumbent matrix, preventing true reproduction of the proven 0.6398 floor and causing consistent -0.0010 train score regressions.

## Why it failed
When dynamically constructing boundaries for `eval_many` (especially from `fm.seed_nae()`), the output format must exactly match the hardcoded dictionary structure (containing `n_field_periods`, `r_cos`, `z_sin`, etc.). Furthermore, copy-pasting or programmatic diffing of long 15-digit float arrays is highly error-prone. If the fallback incumbent is silently typo-corrupted, it will still evaluate cleanly but will structurally regress the guaranteed floor score.

## Verdict
recurring pitfall — NEVER manually edit hardcoded matrix lines if it can be avoided. If constructing dynamic portfolios with NAE/bank seeds, wrap them in a helper that guarantees the exact key schema expected by `eval_many`. Always verify the incumbent score against historical records before expanding the sweep.
