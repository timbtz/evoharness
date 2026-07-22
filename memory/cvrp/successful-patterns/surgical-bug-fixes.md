# Surgical bug fixes to existing code

Diagnosing and fixing a concrete, named defect in the incumbent (or in a just-failed sibling) reliably recovers or gains score; it is the only C-adjacent activity that ever paid off.

## How it was tried
- c0002 (accepted, -0.0906, baseline for the whole run): after c0001's rewrite regressed to -6.39, c0002 diagnosed the exact cause (Clarke-Wright savings loop mutating `i`/`j` during route reversal, plus merges silently skipped when endpoints didn't line up) and rewrote `_savings_routes` to handle all four CW merge orientations with explicit endpoint tracking. Seed c0000 -0.1086 -> -0.0906; this program stayed the effective baseline for 33 generations.
- c0011r1 (repair, accepted, -0.0978): fixed two compile errors from c0011 (`&b1` array-pointer decay where `int*` expected; restore `#include <math.h>`) — mechanical, correct, recovered a runnable candidate.
- c0051 (accepted, -0.0906): fixed the IndexError that killed sibling c0050 (unguarded list access in mutated overlay routes) via bounds checks — recovered validity, though the underlying 2-opt* idea was still worthless.

## Why it worked
- These changes have a falsifiable target (a specific broken line/loop), so the edit is small and its effect is predictable — unlike "improvement" ideas whose effect drowns in eval noise (see performance-analysis/score-noise-and-gate.md).
- Counter-example that bounds the pattern: c0018r1 repaired c0018's compile errors correctly (missing `pos_out`/`seg_in` helpers, missing math.h, `lu`->`plu` typo) but scored -6.12 — a repair only restores compilability; it cannot rescue a bad underlying algorithm (see ineffective-approaches/c-kernel-rewrites.md).

## Verdict
promising. When a candidate fails with a concrete error, one repair attempt is worth its cost. Bug-hunting in the incumbent's Python layer is the highest-expected-value "safe" move left; but do not expect repairs of rewrites to salvage the rewrite.
