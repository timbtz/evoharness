"""Euclidean TSP task: evolve build_tour(dist); a fixed 2-opt polish levels the field."""
from __future__ import annotations

from pathlib import Path

from core.candidate import EvalResult
from tasks.common import run_harness

# Generated instances: uniform points in the unit square, spec = (n, seed).
_GEN = {
    "train": [(50, s) for s in range(101, 105)],
    "val": [(100, s) for s in range(201, 204)],
    "public": [(50, 301), (50, 302), (100, 303), (100, 304), (200, 305), (200, 306)],
}
_TIMEOUT = {"train": 60.0, "val": 60.0, "public": 90.0, "private": 120.0}
# TSPLIB private split (EUC_2D nint distances): name -> known optimal tour length.
_OPTIMA = {"eil51": 426, "berlin52": 7542, "st70": 675, "kroA100": 21282, "ch150": 6528}
_INST_DIR = Path(__file__).parent / "instances"

_TEMPLATE = '''import json, traceback
import numpy as np

def _gen(n, seed): return np.random.default_rng(seed).random((n, 2))

def _dmat(pts, rounded):  # rounded = TSPLIB EUC_2D nint
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    return np.floor(d + 0.5) if rounded else d

def _length(t, d): return float(d[t, np.roll(t, -1)].sum())

def _polish(t, d):  # fixed 2-opt, first improvement, at most 3 full passes
    t, n = list(t), len(t)
    for _ in range(3):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i else n - 1):
                a, b, c, e = t[i], t[i + 1], t[j], t[(j + 1) % n]
                if d[a, c] + d[b, e] < d[a, b] + d[c, e] - 1e-9:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return t

def _nn(d):
    tour, todo = [0], set(range(1, len(d)))
    while todo:
        tour.append(min(todo, key=lambda j: d[tour[-1], j])); todo.discard(tour[-1])
    return tour

def _candidate(d):
    t = [int(v) for v in build_tour(d.copy())]
    if sorted(t) != list(range(len(d))):
        raise ValueError("build_tour must return a permutation of range(%d)" % len(d))
    return _polish(t, d)

try:
    exec(compile(__CODE__, "<candidate>", "exec"), globals())
__BODY__
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)
    loc = " [line %s: %s]" % (tb[-1].lineno, tb[-1].line) if tb else ""
    print(json.dumps({"error": "%s: %s%s" % (type(e).__name__, e, loc)}))
'''

# Spec per instance: ("gen", n, seed) or ("real", points, optimum).
_EVAL_BODY = '''    gaps, lens = [], []
    for spec in __SPECS__:
        if spec[0] == "gen":
            d = _dmat(_gen(spec[1], spec[2]), False)
            ref = _length(_polish(_nn(d), d), d)
        else:
            d, ref = _dmat(np.array(spec[1], float), True), float(spec[2])
        L = _length(_candidate(d), d)
        gaps.append(100.0 * (L - ref) / ref); lens.append(L)
    m = float(np.mean(gaps))
    print(json.dumps({"score": -m, "metrics": {"mean_gap_pct": round(m, 3),
        "gaps_pct": [round(g, 2) for g in gaps],
        "tour_len_mean": round(float(np.mean(lens)), 2)}}))'''

_RENDER_BODY = '''    pts = _gen(50, 101)
    t = _candidate(_dmat(pts, False))
    print(json.dumps({"score": 0.0, "metrics": {
        "points": [[round(float(x), 4), round(float(y), 4)] for x, y in pts],
        "tour": t}}))'''


def _script(code: str, body: str, specs=()) -> str:
    body = body.replace("__SPECS__", repr(list(specs)))
    return _TEMPLATE.replace("__CODE__", repr(code)).replace("__BODY__", body)


def _load_tsp(name: str) -> list[tuple[float, float]]:
    path = _INST_DIR / f"{name}.tsp"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run python tasks/tsp/fetch.py` to download it")
    pts, in_coords = [], False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            in_coords = True
        elif in_coords:
            if line == "EOF" or not line:
                break
            _, x, y = line.split()[:3]; pts.append((float(x), float(y)))
    return pts


def _specs(split: str) -> list[tuple]:
    if split == "private":
        return [("real", _load_tsp(n), opt) for n, opt in _OPTIMA.items()]
    return [("gen", n, s) for n, s in _GEN[split]]


class _TspTask:
    name = "tsp"
    wiki_dir = Path(__file__).parent / "wiki"
    description = """\
Euclidean TSP construction. Given a full symmetric distance matrix, build a
good closed tour. Write exactly this function (numpy is available, already
imported as np in your module):

    def build_tour(dist: np.ndarray) -> list[int]

`dist` is (n, n), symmetric, zero diagonal. Return a permutation of range(n)
— the visiting order; any rotation or direction is fine. Anything else is an
error. The harness then applies the SAME fixed 2-opt polish to every tour
(first-improvement, at most 3 full passes) before measuring closed length,
so cheap local wiggles are already handled: evolve better CONSTRUCTIONS
whose polished result is short, not a 2-opt reimplementation.

Score = -mean(gap_pct); gap_pct = 100 * (len - ref) / ref per instance,
where ref is a nearest-neighbor-from-node-0 tour given the same polish
(the true optimum on the hidden TSPLIB split). Higher is better; the seed
IS nearest neighbor, so it scores ~0 on train. Instances: uniform random
points in the unit square, n = 50-200 (hidden: TSPLIB, integer distances).

Fruitful directions: greedy edge construction, cheapest/farthest insertion,
Clarke-Wright savings, Or-opt segment moves in your own loop, seeded
randomized restarts (np.random.default_rng — keep runs deterministic). A
split must finish in 60-90 s: stay ~O(n^2 log n), vectorize with numpy."""

    def seed_code(self) -> str:
        return """\
import numpy as np

def build_tour(dist):
    n = len(dist)
    tour, todo = [0], set(range(1, n))
    while todo:
        nxt = min(todo, key=lambda j: dist[tour[-1], j])
        tour.append(nxt)
        todo.discard(nxt)
    return tour"""

    def evaluate(self, code: str, split: str) -> EvalResult:
        if split not in _TIMEOUT:
            return EvalResult(float("-inf"), error=f"unknown split: {split}")
        return run_harness(_script(code, _EVAL_BODY, _specs(split)), timeout=_TIMEOUT[split])

    def render(self, code: str, result: EvalResult) -> dict:
        r = run_harness(_script(code, _RENDER_BODY), timeout=60.0)
        if r.error or not r.metrics.get("tour"):
            return {"kind": "tour", "points": [], "tour": []}
        return {"kind": "tour", "points": r.metrics["points"], "tour": r.metrics["tour"],
                "mean_gap_pct": result.metrics.get("mean_gap_pct")}


TASK = _TspTask()
