# Canvas Cropping for Cheap Evals (resolution landmine)
Slicing `r_cos`/`z_sin` down to the minimal nonzero mode support after truncation, to cut eval cost (~20 s → ~2 s). REFUTED: VMEC resolution follows the matrix shape at EVERY fidelity, so a cropped boundary is evaluated at lower resolution everywhere — train and the LF gate happily certify boundaries that val's tighter run rejects.

## How it was tried
Proposed by the in-run analyst (s102 @20 report, its top-ranked P1) with the claim "the boundary is physically unchanged". Four candidates were spent on it in `stellar_p2-s102-48117936`:
- c0012f (val −0.0115): refiner cropped every truncated probe to its minimal canvas.
- c0014 (train 0.6081, val −0.0145): crop cut eval cost from ~20 s to ~2 s and train loved the low-res physics; val collapsed. Its measured `bank_dist 1.07e-3` was also computed against the cropped canvas.
- c0014f (val −0.0109): reverted the polish but kept the crop; same collapse.
- c0015 (val −0.0115): multi-seed truncation ladder on cropped canvases; byte-identical return to c0012f.

## Why it failed
This is the benchmark's documented resolution-gaming failure mode (fidelity-dial.md, paper §resolution gaming): eval resolution is matched to the boundary's matrix shape, so cropping is not a no-op — it lowers the resolution of every subsequent eval INCLUDING the LF verification gate. The cheap-eval "gain" is exactly the mechanism that poisons the returned boundary; there is no fidelity left inside the candidate that can catch it.

## Verdict
refuted — escapes stay on the parent canvas, full stop. There is no safe way to cut the 12–27 s/eval cost of bank-canvas boundaries by reshaping; budget for ~15–25 real evals per episode, or spend cheap evals only on genuinely low-mode seeds (e.g. NAE mp=1 archive arms).
