# Unsafe In-Place Modifications to NAE Seeds
NAE seeds generated via `fm.seed_nae` return delicate nested lists representing Fourier coefficients. When modified in-place (e.g., altering `r[0][ntor]` or slicing/shallow copying rows without returning native `lists`), pydantic validation in `fm.eval` will crash with shape mismatch, ragged arrays, or parse value errors.

## How it was tried (run stellar_p2-s7-30186401)
- c0001 (`-inf`): Replaced `_eval()` return signature `(score, metrics)` but missed loop unpacking in `solve()`, crashing on `TypeError: too many values to unpack`.
- c0004 (`-inf`): Complex `rc[0, ntor] = float(np.asarray(rc[0][ntor]))` logic directly mutated arrays in a way that malformed downstream `rc.tolist()` execution, crashing pydantic.
- c0006 (`-inf`): ValueError: too many values to unpack (expected 2) caused by `(m, b) = fm.eval(b)` while attempting to extract metrics directly alongside scores.

## Why it failed
The `fm.eval` interface in this benchmark strictly returns a single `metrics` dict, which is then passed to `fm.score(m)`. Writers repeatedly assumed `fm.eval` yielded `(score, boundary)` or `(metrics, boundary)` tuples. Furthermore, dynamic type changes from `np.float64` -> python `float` during list comprehensions alter nested list structures in ways that fail `fm`'s pydantic gatekeeper. 

## Verdict
recurring pitfall — Always call `m = fm.eval(b)` and `s = fm.score(m)` sequentially. Never unpack `fm.eval` into multiple items. Use `.tolist()` on the full numpy matrix, do not reconstruct lists element-by-element.
