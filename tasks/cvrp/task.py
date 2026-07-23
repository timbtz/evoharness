"""Anytime CVRP task: evolve a whole solver module (Python policy + optional C
kernels via compile_c) under a fixed per-instance wall-clock budget. Scoring and
route validation happen HOST-side from the routes the sandbox returns, so evolved
code cannot forge a score. Runs in a pinned-CPU docker container when available,
else the unshare sandbox."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from core.candidate import EvalResult
from core.sandbox import docker_image_ready, run_python, run_python_docker
from tasks.common import run_harness

_DIR = Path(__file__).parent
_INST_DIR = _DIR / "instances"
_IMAGE, _DOCKERFILE = "evoharness-cvrp-eval", str(_DIR / "Dockerfile.eval")
_CPUSET = "3" if (os.cpu_count() or 1) >= 4 else None  # least-contended core here

# Scored splits need HEADROOM: the seed solves every small A/B/P instance to
# optimum within budget (gap 0.0 -> saturated gate signal, nothing can ever be
# accepted), so X instances carry train/val and the small ones are demoted to
# the cheap every-5-gens public progress report.
_SPLITS = {
    "train": ["X-n101-k25", "X-n110-k13", "X-n125-k30", "X-n186-k15"],
    # val must match private's STRUCTURE, not just size: customers-per-route (n/k)
    # governs which mechanisms pay off (c0024 vs c0035 head-to-head, 2026-07-23).
    # private n/k = 5.6/9.0/14.4 -> val n/k = 7.0/9.9/5.0 at n=153-242
    "val": ["X-n153-k22", "X-n228-k23", "X-n242-k48"],
    "public": ["A-n32-k5", "A-n45-k6", "A-n60-k9", "B-n50-k7", "P-n55-k10"],
    "private": ["X-n200-k36", "X-n251-k28", "X-n303-k21"],
}
_BUDGET = {"train": 5.0, "val": 5.0, "public": 3.0, "private": 6.0}  # s/instance
# anytime-generalization probe: same val instances at 30s — a second Pareto axis
# (good-at-6s vs good-at-30s), paid only by accepted candidates (loop extra_splits)
_SPLITS["val30"] = _SPLITS["val"]
_BUDGET["val30"] = 30.0
_SLACK = 15.0  # script-level hard-kill headroom: numpy import, gcc, JSON I/O
_WALL_GRACE = 8.0  # host-side: max honest overhead beyond sum of budgets

# Best-known costs, CVRPLIB rounded-euclidean convention. All proven optima
# except X-n303-k21 (BKS). Verified 2026-07-18: CVRPLIB tables + .sol files
# (galgos.inf.puc-rio.br/cvrplib), PyVRP/Instances .sol, neo.lcc.uma.es/vrp;
# .sol costs reproduced exactly under nint rounding.
_BKS = {
    "A-n32-k5": 784, "A-n45-k6": 944, "A-n60-k9": 1354, "B-n50-k7": 741,
    "P-n55-k10": 694, "A-n37-k6": 949, "B-n45-k5": 751, "P-n65-k10": 792,
    "X-n101-k25": 27591, "X-n110-k13": 14971, "X-n125-k30": 55539,
    "X-n153-k22": 21220, "X-n200-k36": 58578, "X-n251-k28": 38684,
    "X-n303-k21": 21736, "X-n176-k26": 47812, "X-n186-k15": 24145,
    "X-n214-k11": 10856, "X-n228-k23": 25742, "X-n242-k48": 82751,
}

_TEMPLATE = '''import ctypes, json, os, shutil, subprocess, sys, tempfile, time, traceback
import numpy as np
_now = time.monotonic  # best-effort only: in-process timing is spoofable, so the
# HOST also bounds total wall-clock (sum of budgets + grace) and re-scores routes

def compile_c(src):
    d = tempfile.mkdtemp(prefix="cvrp_c_")
    cpath, so = os.path.join(d, "k.c"), os.path.join(d, "k.so")
    open(cpath, "w").write(src)
    r = subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC",
                        "-o", so, cpath, "-lm"], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError("C compile failed:\\n" + r.stderr[-1500:])
    lib = ctypes.CDLL(so)
    shutil.rmtree(d, ignore_errors=True)  # .so stays mapped; no /tmp leak on fallback
    return lib

def _dist(coords):  # CVRPLIB EUC_2D: nint-rounded euclidean
    d = np.sqrt(((coords[:, None] - coords[None]) ** 2).sum(-1))
    return np.ascontiguousarray(np.floor(d + 0.5).astype(np.int32))

try:
    exec(compile(__CODE__, "<candidate>", "exec"), globals())
    out = []
    for idx, inst in enumerate(__SPECS__):
        coords = np.array(inst["coords"], float)
        demand = np.array(inst["demand"], np.int32)
        t0 = _now()
        routes = solve(coords, _dist(coords), demand, int(inst["capacity"]),
                       t0 + inst["budget"], compile_c)
        el = _now() - t0
        if el > inst["budget"] + 1.0:
            print(json.dumps({"error": "deadline exceeded on instance %d: used %.2fs of %.1fs (+1s grace)"
                              % (idx, el, inst["budget"])})); sys.exit(0)
        out.append({"seconds": round(el, 2),  # positional: host zips with its own names
                    "routes": [[int(c) for c in r] for r in routes if len(r)]})
    print(json.dumps({"score": 0.0, "metrics": {"solutions": out}}))
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)
    loc = " [line %s: %s]" % (tb[-1].lineno, tb[-1].line) if tb else ""
    print(json.dumps({"error": "%s: %s%s" % (type(e).__name__, e, loc)}))
'''


def _runner(script: str, timeout: float = 30.0, mem_mb: int = 1024):
    if docker_image_ready(_IMAGE, _DOCKERFILE, str(_DIR)):
        res = run_python_docker(script, timeout, mem_mb, image=_IMAGE, cpuset=_CPUSET)
        if "Unable to find image" in (res.stderr or ""):
            # image pruned externally mid-run (9 false -inf in cvrp-s5-54146615):
            # drop the ready-cache, rebuild lazily, retry once
            from core import sandbox
            sandbox._DOCKER_READY.pop(_IMAGE, None)
            if docker_image_ready(_IMAGE, _DOCKERFILE, str(_DIR)):
                res = run_python_docker(script, timeout, mem_mb, image=_IMAGE, cpuset=_CPUSET)
        return res
    return run_python(script, timeout, mem_mb)  # CI / no-docker fallback


def _load_vrp(name: str) -> dict:
    path = _INST_DIR / f"{name}.vrp"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run python tasks/cvrp/fetch.py` first")
    cap, coords, demand, section = 0, {}, {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line == "EOF":
            continue
        if line.endswith("_SECTION"):
            section = line
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            if key.strip() == "CAPACITY":
                cap = int(val)
            continue
        parts = line.split()
        if section == "NODE_COORD_SECTION":
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND_SECTION":
            demand[int(parts[0])] = int(parts[1])
    n = len(coords)
    assert n and cap and demand.get(1) == 0, f"bad .vrp parse: {name}"
    return {"name": name, "capacity": cap, "n": n,
            "coords": [list(coords[i]) for i in range(1, n + 1)],   # node 1 = depot 0
            "demand": [demand[i] for i in range(1, n + 1)], "bks": _BKS[name]}


_CACHE: dict[str, dict] = {}


def _inst(name: str) -> dict:
    if name not in _CACHE:
        _CACHE[name] = _load_vrp(name)
    return _CACHE[name]


def _dmat(inst: dict) -> np.ndarray:
    c = np.array(inst["coords"], float)
    return np.floor(np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)) + 0.5).astype(np.int64)


def _validate_and_cost(inst: dict, routes) -> tuple[str | None, int]:
    n, dem = inst["n"], inst["demand"]
    if not isinstance(routes, list) or not all(isinstance(r, list) for r in routes):
        return "solve() must return a list of routes (lists of customer indices)", 0
    routes = [r for r in routes if r]
    if sum(len(r) for r in routes) > 4 * n:
        return "routes contain far more entries than customers", 0
    flat = [c for r in routes for c in r]
    if any(not isinstance(c, int) or not 1 <= c < n for c in flat):
        return "customer indices must be ints in 1..n-1 (depot 0 must not appear)", 0
    if sorted(flat) != list(range(1, n)):
        miss = len(set(range(1, n)) - set(flat))
        return f"every customer exactly once: {miss} missing, {len(flat) - len(set(flat))} duplicated", 0
    for r in routes:
        load = sum(dem[c] for c in r)
        if load > inst["capacity"]:
            return f"route load {load} exceeds capacity {inst['capacity']}", 0
    d = _dmat(inst)
    return None, int(sum(d[0, r[0]] + d[r, np.roll(r, -1)][:-1].sum() + d[r[-1], 0]
                         for r in map(list, routes)))


def _score(names: list[str], res: EvalResult) -> EvalResult:
    raw = res.metrics.get("solutions")  # positional: solutions[i] answers names[i]
    if not isinstance(raw, list) or len(raw) != len(names) \
            or not all(isinstance(s, dict) for s in raw):
        return EvalResult(float("-inf"),
                          error=f"expected one solution dict per instance ({len(names)})")
    gaps, costs, secs = [], [], []
    for name, s in zip(names, raw):
        inst = _inst(name)
        err, cost = _validate_and_cost(inst, s.get("routes"))
        if err:
            return EvalResult(float("-inf"), error=f"{name}: {err}")
        gaps.append(100.0 * (cost - inst["bks"]) / inst["bks"])
        costs.append(cost); secs.append(s.get("seconds"))
    m = float(np.mean(gaps))
    return EvalResult(-m, metrics={"mean_gap_pct": round(m, 3),
                                   "gaps_pct": [round(g, 2) for g in gaps],
                                   "costs": costs, "seconds": secs},
                      seconds=res.seconds)


class _CvrpTask:
    name = "cvrp"
    wiki_dir = _DIR / "wiki"
    description = (_DIR / "description.md").read_text()
    # anytime eval noise (score units = gap pct pts): same code re-run varies with
    # wall-clock slicing; gates re-run near-ties this close and take the median
    # widened 2026-07-23: semantically identical code swung train -0.0906->-0.0072
    # (X-n125 55690/55551 bimodal attractor, wall-clock-coupled SA) — the gate must
    # median-of-3 any margin in that band, or lucky rolls become incumbents
    noise = {"train": 0.10, "val": 0.15}
    extra_splits = ("val30",)  # scored for accepted candidates only (see core/loop.py)

    def seed_code(self) -> str:
        from tasks.cvrp.seed import seed_code  # assembler: seed/kernel.c + seed/solver.py
        return seed_code()

    def evaluate(self, code: str, split: str) -> EvalResult:
        if split not in _SPLITS:
            return EvalResult(float("-inf"), error=f"unknown split: {split}")
        try:  # specs carry no instance names: deny trivial lookup keys in-sandbox
            names = _SPLITS[split]
            specs = [{**{k: _inst(n)[k] for k in ("coords", "demand", "capacity")},
                      "budget": _BUDGET[split]} for n in names]
        except FileNotFoundError as e:
            return EvalResult(float("-inf"), error=str(e))
        script = _TEMPLATE.replace("__CODE__", repr(code)).replace(
            "__SPECS__", json.dumps(specs))
        res = run_harness(script, timeout=len(names) * _BUDGET[split] + _SLACK,
                          runner=_runner)
        if res.error:
            return res
        if res.seconds > len(names) * _BUDGET[split] + _WALL_GRACE:
            return EvalResult(float("-inf"), seconds=res.seconds,
                              error=f"wall clock {res.seconds:.1f}s exceeds "
                                    f"{len(names)}x{_BUDGET[split]}s + {_WALL_GRACE}s grace")
        try:
            return _score(names, res)
        except Exception as e:  # crafted payloads must score -inf, never kill the run
            return EvalResult(float("-inf"), error=f"scoring failed: {e}"[:300])

    def render(self, code: str, result: EvalResult) -> dict:
        try:
            inst = _inst("A-n32-k5")
        except FileNotFoundError:
            return {"kind": "routes", "points": [], "routes": []}
        spec = [{**{k: inst[k] for k in ("coords", "demand", "capacity")},
                 "budget": 2.0}]
        script = _TEMPLATE.replace("__CODE__", repr(code)).replace(
            "__SPECS__", json.dumps(spec))
        r = run_harness(script, timeout=2.0 + _SLACK, runner=_runner)
        sols = r.metrics.get("solutions") if not r.error else None
        if not (isinstance(sols, list) and sols and isinstance(sols[0], dict)
                and isinstance(sols[0].get("routes"), list)):
            return {"kind": "routes", "points": [], "routes": []}
        return {"kind": "routes", "points": inst["coords"], "routes": sols[0]["routes"],
                "mean_gap_pct": (result.metrics or {}).get("mean_gap_pct")}


TASK = _CvrpTask()
