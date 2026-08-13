# g3 run: Pipeline Restructures, Crashes, and Pathological Selections
In run `g3`, attempts to restructure the evaluation pipeline by removing phases or aggressively deleting fallbacks triggered catastrophic validation crashes (-1.0873 train) or selected pathological boundaries that collapsed in validation.

## How it was tried
- `g3` c0004 (ERR): Attempted to replace the exhausted static grid with a center-of-mass relocation pipeline. Crashed with `NameError: name '_relocate_com' is not defined`.
- `g3` c0008 (REJ, train -1.0886): Replaced the scattered portfolio with a focused, fully batched single-seed contraction sweep. Selected a pathological boundary that crashed validation.
- `g3` c0009 (REJ, train -1.0873): Replaced the pipeline with a 3-stage composed contraction sweep on the B3 anchor. Selected a pathological boundary crashing validation.
- `g3` c0020 (REJ, train -1.0873): Attempted to apply the proven contraction to high-scoring `seed_bank(i)` entries. Selected a pathological boundary that crashed validation.

## Why it failed
Writers assumed that streamlining the pipeline into single dense batches would map the search space more efficiently. The code did exactly this by removing standard fallback phases. However, removing the multi-seed portfolio and its redundant validations left the solver completely blind to VMEC landmines. Without the safety net of the original scattered portfolio, the acceptance key selected heavily contracted but spectrally deformed boundaries that immediately failed physics constraints.

## Verdict
refuted — Never remove the robust multi-seed fallback portfolio in favor of a single-seed dense sweep. The single-seed sweeps structurally betray the non-regression safety net, frequently selecting pathological boundaries that crash validation with -1.0873 train scores.
