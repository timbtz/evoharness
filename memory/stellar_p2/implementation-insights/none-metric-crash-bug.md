# The None-Metric Division Crash
Dynamically evaluating NAE/bank seeds often yields `None` for metrics like `honest_score` or `p2_score`. Passing these directly into mathematical acceptance keys triggers a fatal `TypeError`.

## How it was tried
- `stellar_p2-s204-63425638` c0002, c0004, c0010 (ERR, score -inf): Pivoted to dynamically generated `nfp=2` NAE seeds. The code evaluated them and passed the metrics directly to: `h = metrics.get("honest_score", metrics.get("p2_score", fm.score(metrics)))`. Because the boundary failed physics gates completely, these keys were `None`, causing `bd = ... / bd` or similar downstream divisions to crash with `TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'`. 
- `stellar_p2-s204-63425638` c0005, c0005f (ACC/REJ): Fixed the crash by implementing strict None-guards (`if h is None: return -1e9`) before any mathematical operations were performed.

## Why it failed
The eval pipeline does not guarantee fully populated dictionaries for catastrophic VMEC failures (often returning `{}` or `None` for score components). Naively assuming metrics will resolve to a float breaks the entire batch evaluation.

## Verdict
recurring pitfall — ALWAYS strictly type-check metric values (`if h is None: return -1e9`) before performing float arithmetic in `_accept_key`. Treat missing scores as immediate hard rejects (-1e9).
