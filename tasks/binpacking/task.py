"""Online bin packing task (FunSearch convention): evolve priority(item, bins)."""
from __future__ import annotations

from pathlib import Path

from core.candidate import EvalResult
from tasks.common import run_harness

# Instance specs: (weibull_shape, scale, num_items, seed); shape 0 = uniform ints 20..70.
_SPLITS = {
    "train": [(3, 45, 1000, s) for s in range(2024, 2029)],
    "val": [(3, 45, 1000, s) for s in range(3024, 3029)],
    "public": [(3, 45, 5000, s) for s in range(4024, 4029)],
    "private": [(1.5, 30, 5000, 5024), (1.5, 30, 5000, 5025),
                (0, 0, 5000, 5026), (0, 0, 5000, 5027)],
}

_TEMPLATE = '''import json, math, traceback
import numpy as np

def _gen(sh, sc, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.integers(20, 71, n) if sh == 0 else rng.weibull(sh, n) * sc
    return np.clip(np.round(x), 1, 100).astype(float)

def _pack(items):
    bins = np.full(len(items), 100.0)
    for item in items:
        valid = np.nonzero(bins - item >= 0)[0]
        bins[valid[np.argmax(priority(item, bins[valid]))]] -= item
    return bins

try:
    exec(compile(__CODE__, "<candidate>", "exec"), globals())
__BODY__
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)
    loc = " [line %s: %s]" % (tb[-1].lineno, tb[-1].line) if tb else ""
    print(json.dumps({"error": "%s: %s%s" % (type(e).__name__, e, loc)}))
'''

_EVAL_BODY = '''    ex, used, l1s = [], [], []
    for sh, sc, n, seed in __SPECS__:
        items = _gen(sh, sc, n, seed)
        bins = _pack(items)
        u, l1 = int((bins != 100.0).sum()), math.ceil(items.sum() / 100)
        ex.append((u - l1) / l1); used.append(u); l1s.append(l1)
    m = float(np.mean(ex))
    print(json.dumps({"score": -m, "metrics": {"excess_pct": round(m * 100, 3),
        "avg_bins": round(float(np.mean(used)), 2), "l1_mean": round(float(np.mean(l1s)), 2)}}))'''

_RENDER_BODY = '''    bins = _pack(_gen(3, 45, 300, 2024))
    loads = [round(100.0 - b, 2) for b in bins if b != 100.0][:80]
    print(json.dumps({"score": 0.0, "metrics": {"loads": loads}}))'''


def _script(code: str, body: str, specs=()) -> str:
    body = body.replace("__SPECS__", repr(list(specs)))
    return _TEMPLATE.replace("__CODE__", repr(code)).replace("__BODY__", body)


class _BinPackingTask:
    name = "binpacking"
    wiki_dir = Path(__file__).parent / "wiki"
    description = """\
Online bin packing. Items arrive one at a time and must be placed immediately
into a bin of capacity 100; placements are final. Write exactly this function
(numpy is available, already imported as np in your module):

    def priority(item: float, bins: np.ndarray) -> np.ndarray

`bins` contains the remaining capacities of the FEASIBLE bins only (every entry
satisfies bins - item >= 0). Return one score per bin; the item goes into the
bin with the highest score (argmax). A fresh, empty bin (remaining == 100) is
always among the candidates, so scoring near-empty bins low delays opening new
bins.

Items are drawn from a Weibull(shape=3, scale=45) distribution, rounded to
integers in [1, 100]. Objective: minimize bins used, measured as the fractional
excess over the L1 lower bound ceil(sum(items) / 100) averaged over instances;
score = -mean(excess), higher is better (0 would be optimal). Best-fit, i.e.
`return -(bins - item)`, scores about -0.04 (~4% excess); evolved heuristics
reach under 1%. Fruitful ideas: strong bonuses for near-perfect fits, penalties
for leaving awkward mid-sized residual capacity, nonlinear transforms of the
gap bins - item. Vectorize with numpy (no Python loops over bins); the function
is called once per item on instances of up to 5000 items."""

    def seed_code(self) -> str:
        return "import numpy as np\ndef priority(item, bins):\n    return -(bins - item)"

    def evaluate(self, code: str, split: str) -> EvalResult:
        specs = _SPLITS.get(split)
        if specs is None:
            return EvalResult(float("-inf"), error=f"unknown split: {split}")
        timeout = 120.0 if any(n == 5000 for _, _, n, _ in specs) else 60.0
        return run_harness(_script(code, _EVAL_BODY, specs), timeout=timeout)

    def render(self, code: str, result: EvalResult) -> dict:
        r = run_harness(_script(code, _RENDER_BODY), timeout=30.0)
        loads = r.metrics.get("loads") or []
        if r.error or not loads:
            return {"kind": "bins", "capacity": 100, "loads": []}
        return {"kind": "bins", "capacity": 100, "loads": loads,
                "excess_pct": result.metrics.get("excess_pct")}


TASK = _BinPackingTask()
