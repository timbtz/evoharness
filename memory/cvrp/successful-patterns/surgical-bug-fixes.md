# Surgical bug fixes to existing code

Diagnosing and fixing a concrete, named defect in the incumbent reliably recovers or gains score; it is the only C-adjacent activity that ever paid off.

## How it was tried
- c0002 (cvrp-s11-66566581): writer tried to implement demand-urgency tie-breaks and multi-restart, but the actual accepted code was a surgical O(1) `route_of` array replacement for `for k in range(nr)` loops.
- c0011r1 (cvrp-s3-45465035): fixed compile errors (`&b1` decay, `math.h`).
- c0050/c0051 (cvrp-s5-54146615): fixed IndexError via bounds checks.
- c0057 (cvrp-s5-54146615): fixed O(n) `list.index()` lookups by tracking position.
- c0013 (cvrp-s11-66566581): included a surgical fix for a `try_intra_3opt` bug (reversing the wrong segment / degenerate gain calculation) alongside an untested perturbation wrapper.

## Why it worked
- These changes have a falsifiable target (a specific broken line/loop), so the edit is small and its effect is predictable — unlike "improvement" ideas whose effect drowns in eval noise (see performance-analysis/score-noise-and-gate.md).
- Counter-example that bounds the pattern: c0018r1 repaired c0018's compile errors correctly but scored -6.12 — a repair only restores compilability; it cannot rescue a bad underlying algorithm (see ineffective-approaches/c-kernel-rewrites.md).

## Verdict
promising. When a candidate fails with a concrete error, one repair attempt is worth its cost. Bug-hunting in the incumbent's Python layer is the highest-expected-value "safe" move left.
