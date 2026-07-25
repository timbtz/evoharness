# Low-Fidelity Verification and Calibration Gates
Using `fm.eval(b, fidelity="low_fidelity")` (lf) inside `solve()` provides a ground-truth check against the train (vlf) to lf gap that destroys naive escape attempts. Costs ~2.2s per eval. Strictly necessary for any candidate attempting structural novelty escapes.

## How it was tried
- `stellar_p2-s100-89908732` c0008f: Replaced binary `_qi_ok` with a continuous QI-margin penalty. Added an explicit 4-eval reserved budget at the end of `solve()` to evaluate top finalists at lf and return the best lf-survivor. **SUCCESS:** Val flipped from -0.0119 to +0.5913. OFFICIAL 0.6286 — the only new-best of branch B1.
- `stellar_p2-s100-89908732` c0010f: Attempted to dynamically calibrate the `SAFE` margins by firing one mid-run lf probe to measure the vlf->lf feasibility gap. Scored 0.5650/0.5697 (worse).
- `stellar_p2-s100-89908732` c0011f, c0013f, etc.: Integrated 2-3 eval lf finalist gates into their novelty escape mechanisms, preventing vlf->lf fidelity collapse and achieving positive val scores (0.586 - 0.591).
- `stellar_p2-s101-20239089` c0001, c0001f: Relied completely on the 4-eval LF finalist gate to survive the vlf->lf gap after structural mode-truncation escape. c0001f fixed the LF gate's selection key from raw `fm.score` to the novelty-aware harness key, protecting the escaped boundary from being silently replaced by the in-ball anchor.

## Why it worked / failed
The continuous penalty forces the search to trade raw L for QI margin. The lf verification gate guarantees no untested boundary is returned. Escapes survive because the LF gate filters out candidates that cross the ball but secretly break physics.

## Verdict
promising — The lf verification gate on finalists is strictly necessary for any candidate attempting novelty escapes. Continuous QI penalties are superior to binary gates (binary bars in wrong units silently disable, see implementation-insights/qi-safe-units-bug.md).
