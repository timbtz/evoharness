# Crash-classifier guards
fm.eval failures are informative, not noise: classify fm.last_error and react instead of blindly re-mutating.

## Known failure strings (each cost one budget unit when hit)
- "Minimum of B is at the boundary." — boundary shape rejected by the QI machinery; typically over-aggressive mutation. Shrink sigma / revert.
- "VMEC++ did not converge" — equilibrium solver gave up; often high modes or self-intersecting-ish surfaces. Back off the last inflation or step.
- pydantic validation errors — malformed matrices (shape mismatch after a buggy pad/reshape).

## Untested reactions
- Pre-eval sanity screen (finite coeffs, r_cos[0][ntor] dominant, |high modes| << minor radius) to avoid spending budget on guaranteed crashes — a free geometry-only filter.
- Crash-aware step-size: halve sigma after a crash, restore after k successes.
- Crash streak (3+) -> restart from best-known-good.

## Verdict
promising, untested — cheap logic, directly saves budget units.
