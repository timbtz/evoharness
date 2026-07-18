# Pitfalls: how candidates fail validation or time out

## Validator facts (read these literally)
- Output must be exactly shape (26, 3), rows (x, y, r), convertible to float.
- Checks, each with tolerance 1e-9: containment `x-r >= -tol`, `x+r <= 1+tol`
  (same for y); pairwise `dist >= r_i + r_j - tol`; `r >= 0`; no NaN.
- Any violation → score -inf with an error naming the offending circle(s),
  e.g. "circles 3 and 7 overlap: dist=... < r3+r7=...".

## Top failure modes
1. **Float-dust overlap**: radii computed to *exactly* touch (`r_i + r_j == d`)
   fail when rounding pushes the sum 1e-12 over. Fix: after all optimization,
   `r = np.maximum(r - 1e-10, 0.0)` or `r *= 1.0 - 1e-9`. Costs ~2.6e-9 of
   score; buys validity.
2. **Containment off-by-epsilon**: clipping centers to [0,1] but setting
   r = wall distance computed *before* the clip. Recompute walls from final
   centers; shrink as above.
3. **NaN from zero distance**: two centers coincide (bad init or a teleport
   collision) → `diff/d` divides by 0 → NaN centers → "NaN at row i".
   Guard: `d = np.maximum(d, 1e-12)` and never place two circles at the same
   point.
4. **Negative radii**: over-aggressive shrinking or `d_ij - r_j` going
   negative in the max-radii sweep. Always `np.clip(r, 0, None)` last.
5. **scipy import**: not installed. `from scipy.optimize import minimize`
   dies at import time → instant error. Pure numpy only (math/itertools ok).
6. **Nondeterminism**: `np.random.rand()` / unseeded `default_rng()` makes
   scores irreproducible across evaluations. Always `default_rng(0)` (any
   fixed seed). Time-based loops (`while time.time() < t0+55`) are also
   nondeterministic in effect — prefer fixed iteration counts.
7. **Timeout (60 s)**: nested Python loops per iteration per pair with large
   iteration counts, or an optimizer that doesn't terminate. n=26 is tiny —
   vectorize pair math; if you must loop over 26 circles, that's fine, but
   don't loop over 325 pairs x 1e5 iterations in pure Python.
8. **Wrong output shape**: returning (centers, radii) tuple, a (26, 2) plus
   separate radii, or shape (3, 26). Must be one (26, 3) ndarray.
9. **Hardcoding n=26 wrongly**: pack(n) receives n=26; using a global n or
   building 30 grid points and forgetting to slice to n gives shape errors.

## Score-losing (valid but weak) habits
- Returning the *last* iterate instead of the *best* seen.
- Uniform radii: caps you near 2.19 no matter how good the centers are.
- Optimizing centers but never re-solving radii (score never moves).
- One global scale factor as the only validity mechanism: it punishes every
  circle for the single worst overlap; resolve pairs locally, scale globally
  only as the final 1e-9 safety net.
- Printing inside pack(): the harness reads the last stdout line as JSON;
  the eval snippet prints last so stray prints usually still parse, but heavy
  printing wastes the time budget. Keep stdout quiet.
