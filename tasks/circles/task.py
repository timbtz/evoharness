"""Circle packing task: pack n=26 circles in the unit square, maximize sum of radii.

Single deterministic instance — train/public/private splits are identical by design
(no instance distribution), so generalization metrics do not apply here.
Validator adapted from openevolve/examples/circle_packing/evaluator.py (Apache-2.0).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.candidate import EvalResult
from tasks.common import run_harness

TIMEOUT = 60.0

DESCRIPTION = """\
Pack n=26 circles inside the unit square [0,1]^2, maximizing the SUM OF RADII.

Write a single function (numpy available, no scipy, no network):

    def pack(n: int) -> np.ndarray

Return a numpy array of shape (n, 3); row i is (x_i, y_i, r_i).
Constraints, each checked with tolerance 1e-9:
- containment: x - r >= 0 and x + r <= 1 (same for y), for every circle
- no overlap: dist(center_i, center_j) >= r_i + r_j for every pair i < j
- all radii >= 0, no NaN values anywhere
Score = sum of radii (higher is better). Best known for n=26 is ~2.635 (AlphaEvolve).

Iterative optimization inside pack() is allowed and encouraged: pure-numpy
gradient/repulsion relaxation, perturb-and-reinflate loops, pattern search, etc.
Time limit is 60 s — budget iterations accordingly (~50k light relaxation steps).
Must be deterministic: no randomness unless seeded (e.g. np.random.default_rng(0)).

Good packings mix radii: large circles in the interior, small ones wedged into
corners and gaps. A useful primitive: place centers, set each radius to its
distance to the nearest wall, then repeatedly scale overlapping pairs by
dist/(r_i + r_j) and re-inflate any remaining slack.
"""

_SEED = '''import numpy as np


def pack(n: int) -> np.ndarray:
    """6x5 grid of circles with a uniform safe radius (valid for n <= 30)."""
    cols, rows = 6, 5
    r = 0.5 / cols - 1e-4  # half the tightest spacing, with margin
    pts = [((i + 0.5) / cols, (j + 0.5) / rows)
           for j in range(rows) for i in range(cols)][:n]
    out = np.zeros((n, 3))
    out[:, :2] = pts
    out[:, 2] = r
    return out
'''

# Appended after the candidate code; prints exactly one JSON line.
_EVAL = """

import json as _json
import numpy as _np


def _validate(a):
    tol = 1e-9
    if a.shape != (26, 3):
        return "expected shape (26, 3), got %s" % (a.shape,)
    if _np.isnan(a).any():
        return "NaN at row %d" % int(_np.where(_np.isnan(a).any(axis=1))[0][0])
    x, y, r = a[:, 0], a[:, 1], a[:, 2]
    if (r < 0).any():
        i = int(_np.argmax(r < 0))
        return "circle %d has negative radius %.6g" % (i, r[i])
    out = (x - r < -tol) | (x + r > 1 + tol) | (y - r < -tol) | (y + r > 1 + tol)
    if out.any():
        i = int(_np.argmax(out))
        return "circle %d (x=%.6f, y=%.6f, r=%.6f) outside unit square" % (i, x[i], y[i], r[i])
    d = _np.sqrt((x[:, None] - x) ** 2 + (y[:, None] - y) ** 2)
    need = r[:, None] + r
    _np.fill_diagonal(need, -1.0)
    bad = d < need - tol
    if bad.any():
        i, j = map(int, _np.argwhere(bad)[0])
        return "circles %d and %d overlap: dist=%.9f < r%d+r%d=%.9f" % (i, j, d[i, j], i, j, need[i, j])
    return None


try:
    _a = _np.asarray(pack(26), dtype=float)
    _err = _validate(_a)
    if _err:
        print(_json.dumps({"error": _err}))
    else:
        _s = float(_a[:, 2].sum())
        print(_json.dumps({"score": _s, "metrics": {
            "sum_radii": _s, "target": 2.635, "ratio": _s / 2.635}}))
except Exception as _e:
    print(_json.dumps({"error": "%s: %s" % (type(_e).__name__, _e)}))
"""

_RENDER = """

import json as _json
import numpy as _np

try:
    _a = _np.asarray(pack(26), dtype=float)
    assert _a.shape == (26, 3) and not _np.isnan(_a).any()
    print(_json.dumps({"score": 0.0, "metrics": {
        "circles": [[round(float(v), 4) for v in row] for row in _a],
        "sum_radii": round(float(_a[:, 2].sum()), 4)}}))
except Exception as _e:
    print(_json.dumps({"error": str(_e)}))
"""


def evaluate(code: str, split: str = "train") -> EvalResult:
    """All splits are the same single deterministic instance."""
    return run_harness(code + _EVAL, timeout=TIMEOUT)


def render(code: str, result: EvalResult) -> dict:
    res = run_harness(code + _RENDER, timeout=TIMEOUT)
    if res.error or "circles" not in res.metrics:
        return {"kind": "circles", "circles": []}
    return {"kind": "circles", "circles": res.metrics["circles"],
            "sum_radii": res.metrics["sum_radii"]}


TASK = SimpleNamespace(
    name="circles",
    description=DESCRIPTION,
    wiki_dir=Path(__file__).parent / "wiki",
    seed_code=lambda: _SEED,
    evaluate=evaluate,
    render=render,
)
