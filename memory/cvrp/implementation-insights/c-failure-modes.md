# How C-kernel changes die

C-touching candidates fail in four distinct ways; distinguishing them matters because only compile errors are reliably repairable, and "ran nice but scored -5" is an algorithm problem, not a code-health problem.

## How it was tried (failure taxonomy of this run)
- Compile errors (repairable): c0011 (`&b1` where `int*` expected — array-pointer decay; `exp()` after deleting `#include <math.h>`), c0018 (deleted `pos_out`/`seg_in` helpers still referenced; `lu` vs `plu` typo). Repairs c0011r1/c0018r1 both restored compilation in one step.
- Heap corruption (crash at runtime): c0020 `munmap_chunk(): invalid pointer` from freeing rows of the 2D `RPRE` prefix-load array after reassigning row pointers. The RPRE ownership scheme is the kernel's most fragile spot.
- Silent no-output (worst: zero diagnostic): c0003, c0005, c0007, c0008, c0039 — big C edits that produced no solution within the budget (hang, infeasible state, or early abort). 5 candidates burned with no error string.
- Compiles-and-runs-but-regresses (NOT a code failure): c0001 (-6.39), c0018r1 (-6.12), c0023 (-6.75), c0033 (-4.96), c0021 (-2.53). These are algorithmic regressions, not build/health problems; do not treat them as fixable by code repair.
- Pre-tracing history: ~36 C-change failures with recurring variants — undeclared vars, missing `math.h` (with `pow`/`exp` use, ~7x), missing `<time.h>`, undefined C-Python ctypes bindings, invalid pointers (`free()`, `munmap_chunk()`), and Python attribute errors.

## Why it happens
- The kernel relies on invariants that are invisible locally: helpers used across functions, `math.h` for `exp`, row-pointer ownership in `RPRE`, feasibility maintained jointly by ruin/recreate/try_moves. Any medium-size edit breaks one.
- Score-level regressions correlate with edit size, not with edit "safety" claims.

## Verdict
promising as a checklist, exhausted as an experiment: keep `#include <math.h>` and `<time.h>`; never delete helpers without grepping call sites; pass `int[3]` arrays bare (decay), never `&arr`; do not touch RPRE allocation; expect one repair round to fix compiles but not algorithms.
