# `honest_score` Hidden API TypeError Crash
Calling `m.get("honest_score", ...)` with a default parameter triggers a fatal `TypeError` if the evaluation metrics dictionary `m` is wrapped in a custom Pydantic model that does not accept default positional arguments for `.get()`.

## How it was tried
- `plan1-deep-s5-85591679` c0005 (ERR, score -inf): Replaced the standard acceptance key with a custom bucketed key `_bucket_key(m, b=None, fm=None)` that called `_honest(m)` to extract the score. The `_honest(m)` function used `m.get("honest_score")`, but the calling scope passed it alongside a `None` positional argument (`_honest(m, None)` or similar mismatch in the fallback chain), crashing immediately with `TypeError: _honest() takes 1 positional argument but 2 were given [line 774: raise self._value]`.

## Why it failed
The harness's evaluation metrics are parsed via strict Pydantic models. If you define a custom wrapper or helper function, any mismatch in positional vs. keyword arguments during error handling or tuple unpacking will trigger a `TypeError` deep within the model's value extraction logic (`raise self._value`).

## Verdict
recurring pitfall — When wrapping metrics in custom helper functions, ALWAYS ensure exact argument matching. Prefer `m.get("key", default)` directly inline rather than passing metrics through multiple helper layers.
