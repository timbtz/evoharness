"""ConStellaration P2 ("simple-to-build QI stellarator"): evolve an OPTIMIZER
program, not a solution. The candidate is a Python module `solve(fm, rng)` that
searches the stellarator-boundary Fourier space through a metered forward-model
handle and returns its best boundary dict. Fitness = P2 score of that boundary.

Anti-forging: train runs candidate code in-container and its score is advisory
selection pressure only; val/public/private re-score ONLY the returned boundary
JSON in a fresh clean-room container run (no candidate code) — the CVRP
"host re-scores routes" pattern, adapted to a 128s forward model via a cache of
returned boundaries keyed by code hash. Truth is exclusively the official
SimpleToBuildQIStellarator().evaluate() at high fidelity (private split).

Score shaping (gate needs gradient below the feasibility wall, DECISIONS.md):
shaped = official P2 score (0..1) if feasible else -max_normalized_violation.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from core.candidate import EvalResult
from core.sandbox import docker_image_ready, run_python_docker

_DIR = Path(__file__).parent
_ROOT = _DIR.parent.parent
_IMAGE, _DOCKERFILE = "evoharness-stellar-eval", str(_DIR / "Dockerfile.eval")
_ARCHIVE = _ROOT / "memory" / "stellar_p2" / "archive.jsonl"

# Splits = fidelity x budget tiers. Deterministic sim (single-threaded VMEC++,
# fixed rng seed) -> no noise/median machinery. Spike timings (2026-07-23):
# very_low+QI 1.4s, low+QI 2.2s, official high-fid ~128s per forward call.
_FIDELITY = {"train": "very_low_fidelity", "val": "low_fidelity",
             "public": "very_low_fidelity"}
_TRAIN = {"max_evals": 72, "cpu_budget": 240.0, "seed": 0, "collect_top": 8,
          "workers": 2,  # box has 4 cores: 2 for the eval container, 2 for host
          "eval_timeout": 60.0}  # hard per-eval kill (mutated high-mode
          # boundaries can hang VMEC near-forever; 60s >> the 27s worst
          # healthy eval observed)
# deep-eval campaigns raise the budget per candidate without a code change:
# STELLAR_TRAIN_OVERRIDES='{"max_evals":160,"cpu_budget":480.0}' on server start
_TRAIN.update(json.loads(os.environ.get("STELLAR_TRAIN_OVERRIDES", "{}")))

# Public leaderboard P2 submissions as optional seed bank (user decision
# 2026-07-24). Results seeded from it are "refined from public submissions",
# never "from scratch" — provenance in the JSON, disclosure in NOTICE.md.
_BANK_FILE = _DIR / "seed_bank.json"
_BANK = json.loads(_BANK_FILE.read_text())["seeds"] if _BANK_FILE.exists() else []
_SLACK = 240.0   # container start + jax import/JIT + the last in-flight eval
# overshooting the CPU deadline + final re-score — high-mode (11,21) boundaries
# take 12-27s per forward call, so every tail item is ~20x the mp=1 case
_MEM_MB = 4096   # jax wants >= 2 GB; fork workers add ~0.5 GB each over COW

# boundaries at least this good (shaped) enter the persistent archive; anything
# beating the archive's current best is always kept so a frontier trail exists
# from the first infeasible runs on
_ARCH_MIN = -0.10

_PRELUDE = '''import json, sys, time, traceback
import numpy as np
_T0 = time.monotonic()
CFG = json.loads(__CFG__)  # JSON string literal: survives null/true/false
from constellaration import forward_model as _fmod, initial_guess as _ig, problems as _problems
from constellaration.geometry import surface_rz_fourier as _srf
from constellaration.mhd import vmec_settings as _vs

_P2 = _problems.SimpleToBuildQIStellarator()

def _settings(fidelity=None):
    return _fmod.ConstellarationSettings(
        vmec_preset_settings=_vs.VmecPresetSettings(
            fidelity=fidelity or CFG["fidelity"]),
        turbulent_settings=None)  # turbulent metrics are not part of P2

_SETTINGS = _settings()

def _assess(m):
    feas = _P2.compute_feasibility(m)
    real = _P2._score(m) if _P2.is_feasible(m) else 0.0
    return feas, real, (real if feas <= 1e-2 else -feas)

def _bdict(b):
    return json.loads(b.model_dump_json())
'''

# ---- optimizer-run template: candidate code + metered fm handle ------------
_TEMPLATE = _PRELUDE + '''
def _worker(job):
    """One forward eval in a pool worker; never raises. job = (boundary, fid)."""
    boundary, fid = job
    try:
        b = _srf.SurfaceRZFourier.model_validate(boundary)
        m, _ = _fmod.forward_model(b, settings=_settings(fid))
    except Exception as e:
        return None, str(e)[:300]
    d = {k: v for k, v in m.model_dump().items() if isinstance(v, (int, float))}
    d["feasibility"], d["p2_score"], d["shaped_score"] = _assess(m)
    d["fidelity"] = fid or CFG["fidelity"]
    return (d, _bdict(b)), None

# ALL evals run in pool workers, forked BEFORE any jax op (fork-after-XLA-init
# deadlocks; workers pay their own one-time ~9s JIT). Pathological boundaries
# can hang VMEC inside C++ where no Python signal can interrupt — the batch
# timeout terminates the whole pool and respawns it (fresh workers re-JIT,
# ~15s: the bounded price of a hang, instead of losing the entire budget).
import math as _math
import multiprocessing as _mp
_CTX = _mp.get_context("fork")
_POOL = _CTX.Pool(CFG["workers"])

def _pool_eval(batch, fidelity=None):
    global _POOL
    limit = CFG["eval_timeout"] * _math.ceil(len(batch) / CFG["workers"]) + 15.0
    if fidelity == "low_fidelity":
        limit *= 2  # tighter force tolerance runs longer
    try:
        return _POOL.map_async(_worker, [(b, fidelity) for b in batch]).get(limit)
    except _mp.TimeoutError:
        _POOL.terminate()
        _POOL.join()
        _POOL = _CTX.Pool(CFG["workers"])
        return [(None, "eval hard-timeout: batch killed after %.0fs" % limit)] * len(batch)

_LOG = []  # (shaped, real, feas, boundary) for every successful metered eval

class _FM:
    """Metered forward-model handle. eval()/eval_many() never raise into
    candidate code; every attempted boundary costs one budget unit."""
    def __init__(self):
        self.used, self.last_error = 0, None
    def remaining(self):
        return CFG["max_evals"] - self.used
    def eval_many(self, boundaries, fidelity=None):
        if fidelity not in (None, "very_low_fidelity", "low_fidelity"):
            self.last_error = "fidelity must be None, 'very_low_fidelity' or 'low_fidelity'"
            return [None] * len(boundaries)
        self.last_error = None
        if time.monotonic() - _T0 > CFG["cpu_budget"]:
            self.used = CFG["max_evals"]
            self.last_error = "cpu budget exhausted"
            return [None] * len(boundaries)
        batch = list(boundaries)[:max(self.remaining(), 0)]
        if not batch:
            self.last_error = "eval budget exhausted"
            return [None] * len(boundaries)
        self.used += len(batch)
        results = _pool_eval(batch, fidelity)
        out = []
        for ok, err in results:
            if ok is None:
                self.last_error = err
                out.append(None)
            else:
                d, bd = ok
                _LOG.append((d["shaped_score"], d["p2_score"], d["feasibility"], bd))
                out.append(d)
        out += [None] * (len(boundaries) - len(batch))
        if len(batch) < len(boundaries):
            self.last_error = "eval budget exhausted"
        return out
    def eval(self, boundary, fidelity=None):
        return self.eval_many([boundary], fidelity)[0]
    def score(self, metrics):
        return float(metrics["shaped_score"]) if metrics else float("-inf")
    def seed_nae(self, **kw):
        return _bdict(_ig.generate_nae(**kw))
    def seed_ellipse(self, **kw):
        return _bdict(_ig.generate_rotating_ellipse(**kw))
    def seed_bank_info(self):
        """Official high-fid scores of the public-submission seed bank."""
        return [{"i": i, "official_score": e["score"],
                 "official_feasibility": e["feasibility"]}
                for i, e in enumerate(CFG["seed_bank"])]
    def seed_bank(self, i):
        return json.loads(json.dumps(CFG["seed_bank"][int(i)]["boundary"]))
    def bank_dist(self, boundary):
        """Max-coefficient distance to the closest same-nfp bank seed (padded
        canvas) — the export guard's + harness novelty penalty's exact metric.
        None if no same-nfp seed exists. Free (no eval budget)."""
        def pad(a, shape):
            out = np.zeros(shape); r, c = a.shape; off = (shape[1] - c) // 2
            out[:r, off:off + c] = a
            return out
        def norm(b):
            # scale-normalize by the boundary's own R0: official metrics are all
            # dimensionless, so a uniform rescale is physics-null and must not
            # count as novelty distance (2026-07-25)
            rc = np.asarray(b["r_cos"], float); zs = np.asarray(b["z_sin"], float)
            s = rc[0, rc.shape[1] // 2]
            s = s if s > 0.1 else 1.0
            return rc / s, zs / s
        rc0, zs0 = norm(boundary)
        best = None
        for e in CFG.get("seed_bank", []):
            bb = e["boundary"]
            if bb.get("n_field_periods") != boundary.get("n_field_periods"):
                continue
            rc1, zs1 = norm(bb)
            shape = (max(rc0.shape[0], rc1.shape[0]), max(rc0.shape[1], rc1.shape[1]))
            d = max(np.abs(pad(rc0, shape) - pad(rc1, shape)).max(),
                    np.abs(pad(zs0, shape) - pad(zs1, shape)).max())
            best = d if best is None else min(best, d)
        return best

try:
    exec(compile(__CODE__, "<candidate>", "exec"), globals())
    _fm = _FM()
    _boundary = solve(_fm, np.random.default_rng(CFG["seed"]))
    # authoritative (uncounted) re-score of the returned artifact via the pool
    # (same hang protection); candidates cannot forge this number
    _ok, _err = _pool_eval([_boundary])[0]
    if _ok is None:
        raise RuntimeError("final re-score of returned boundary failed: %s" % _err)
    _md, _bd = _ok
    _feas, _real, _shaped = _md["feasibility"], _md["p2_score"], _md["shaped_score"]
    _seen, _arch = set(), []
    # loop vars deliberately distinct from _bd: until 2026-07-25 this loop rebound
    # _bd, so the reported "boundary" was the collect_top-th logged eval instead of
    # the returned one — poisoning val/private/best.py for every candidate
    for _as, _ar, _af, _ab in sorted(_LOG, key=lambda t: -t[0]):
        _k = json.dumps(_ab, sort_keys=True)
        if _k not in _seen:
            _seen.add(_k)
            _arch.append({"shaped": _as, "p2": _ar, "feasibility": _af, "boundary": _ab})
        if len(_arch) >= CFG["collect_top"]:
            break
    print(json.dumps({"score": _shaped, "metrics": {
        "p2_score": _real, "feasibility": _feas, "shaped_score": _shaped,
        "objective_L": _md.get("minimum_normalized_magnetic_gradient_scale_length"),
        "aspect_ratio": _md.get("aspect_ratio"),
        "max_elongation": _md.get("max_elongation"),
        "edge_iota_per_nfp": _md.get("edge_rotational_transform_over_n_field_periods"),
        "edge_mirror_ratio": _md.get("edge_magnetic_mirror_ratio"),
        "log10_qi": float(np.log10(_md["qi"])) if _md.get("qi") else None,
        "evals_used": _fm.used, "seconds_used": round(time.monotonic() - _T0, 1),
        "boundary": _bd, "archive": _arch}}))
except Exception as e:
    _tb = traceback.extract_tb(e.__traceback__)
    _loc = " [line %s: %s]" % (_tb[-1].lineno, _tb[-1].line) if _tb else ""
    print(json.dumps({"error": "%s: %s%s" % (type(e).__name__, e, _loc)}))
'''

# ---- clean-room verify template: boundary JSON only, no candidate code -----
_VERIFY = _PRELUDE + '''
try:
    _b = _srf.SurfaceRZFourier.model_validate(CFG["boundary"])
    if CFG["official"]:
        _ev = _P2.evaluate(_b)  # the ONLY number ever reported as truth
        print(json.dumps({"score": _ev.score, "metrics": {
            "p2_score": _ev.score, "feasibility": _ev.feasibility,
            "objective_L": _ev.objective, "official": True}}))
    else:
        _m, _ = _fmod.forward_model(_b, settings=_SETTINGS)
        _feas, _real, _shaped = _assess(_m)
        print(json.dumps({"score": _shaped, "metrics": {
            "p2_score": _real, "feasibility": _feas, "shaped_score": _shaped,
            "objective_L": _m.minimum_normalized_magnetic_gradient_scale_length,
            "max_elongation": _m.max_elongation, "aspect_ratio": _m.aspect_ratio,
            "edge_iota_per_nfp": _m.edge_rotational_transform_over_n_field_periods,
            "edge_mirror_ratio": _m.edge_magnetic_mirror_ratio,
            "log10_qi": float(np.log10(_m.qi)) if _m.qi else None}}))
except Exception as e:
    print(json.dumps({"error": ("%s: %s" % (type(e).__name__, e))[:500]}))
'''


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _bkey(boundary: dict) -> str:
    """Dedupe key: nfp + coefficient matrices rounded to 1e-4."""
    rc = np.round(np.asarray(boundary["r_cos"], float), 4)
    zs = np.round(np.asarray(boundary["z_sin"], float), 4)
    h = hashlib.sha1(rc.tobytes() + zs.tobytes()).hexdigest()[:16]
    return f'{boundary.get("n_field_periods", 1)}-{h}'


_NOVELTY_MIN, _NOVELTY_PEN = 1e-3, 0.05  # export guard refuses < 1e-3 (submit_export)


def _scale_norm(boundary: dict) -> tuple[np.ndarray, np.ndarray]:
    """Coefficient matrices divided by the boundary's own R0 (r_cos[0, n=0]).
    The official metrics are all dimensionless, so a uniform rescale is
    physics-null — without this normalization it would manufacture novelty
    distance for a boundary that is physically a copy (2026-07-25)."""
    rc = np.asarray(boundary["r_cos"], float)
    zs = np.asarray(boundary["z_sin"], float)
    s = rc[0, rc.shape[1] // 2]
    s = s if s > 0.1 else 1.0
    return rc / s, zs / s


def bank_distance(boundary: dict) -> float | None:
    """Max-coefficient distance to the closest same-nfp public seed-bank
    boundary, scale-normalized and padded to a common canvas — the export
    guard's exact metric."""
    if not _BANK:
        return None

    def pad_to(a, shape):
        out = np.zeros(shape)
        r, c = a.shape
        off = (shape[1] - c) // 2
        out[:r, off:off + c] = a
        return out

    rc0, zs0 = _scale_norm(boundary)
    best = None
    for e in _BANK:
        bb = e["boundary"]
        if bb.get("n_field_periods") != boundary.get("n_field_periods"):
            continue
        rc1, zs1 = _scale_norm(bb)
        shape = (max(rc0.shape[0], rc1.shape[0]), max(rc0.shape[1], rc1.shape[1]))
        d = max(np.abs(pad_to(rc0, shape) - pad_to(rc1, shape)).max(),
                np.abs(pad_to(zs0, shape) - pad_to(zs1, shape)).max())
        best = d if best is None else min(best, d)
    return best


def _novelty_shape(res: EvalResult, d: float | None) -> EvalResult:
    """Train/val fitness shaping (novelty decision 2026-07-24): a FEASIBLE
    boundary inside the export guard's 1e-3 ball is a near-copy of a public
    submission — unsubmittable — so its harness fitness pays up to _NOVELTY_PEN,
    linearly decaying to 0 at the ball's edge (gradient points OUT of the ball).
    The official private score is never shaped: it stays the truth."""
    if d is None or res.error or res.score <= 0 or d >= _NOVELTY_MIN:
        return res
    pen = round(_NOVELTY_PEN * (1.0 - d / _NOVELTY_MIN), 4)
    res.metrics["novelty_penalty"] = pen
    return EvalResult(res.score - pen, metrics=res.metrics,
                      error=res.error, seconds=res.seconds)


def _runner(script: str, timeout: float = 30.0, mem_mb: int = _MEM_MB,
            cpus: int = 1):
    if not docker_image_ready(_IMAGE, _DOCKERFILE, str(_DIR)):
        return None  # constellaration only exists in the image: no local fallback
    res = run_python_docker(script, timeout, mem_mb, image=_IMAGE, cpus=cpus)
    if "Unable to find image" in (res.stderr or ""):
        from core import sandbox  # image pruned externally: rebuild once, retry
        sandbox._DOCKER_READY.pop(_IMAGE, None)
        if docker_image_ready(_IMAGE, _DOCKERFILE, str(_DIR)):
            res = run_python_docker(script, timeout, mem_mb, image=_IMAGE, cpus=cpus)
    return res


def verify_boundary(boundary: dict, official: bool = False,
                    fidelity: str = "low_fidelity",
                    timeout: float | None = None) -> EvalResult:
    """Clean-room scoring of one boundary dict (also used by submit_export)."""
    cfg = {"boundary": boundary, "official": official, "fidelity": fidelity}
    script = _VERIFY.replace("__CFG__", repr(json.dumps(cfg)))
    t = timeout or (1800.0 if official else _SLACK)  # high-mode boundaries can
    # take many minutes at official fidelity
    r = _runner(script, timeout=t)
    if r is None:
        return EvalResult(float("-inf"), error="docker image unavailable "
                          f"(build {_IMAGE} from {_DOCKERFILE})")
    return _parse(r, t)


def _parse(r, timeout: float) -> EvalResult:
    """run_harness's contract, applied to an already-executed SandboxResult."""
    if r.timed_out:
        return EvalResult(float("-inf"), error=f"timeout after {timeout}s",
                          seconds=r.seconds)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    try:
        payload = json.loads(lines[-1])
        assert isinstance(payload, dict)
    except (IndexError, ValueError, AssertionError):
        err = (r.stderr or r.stdout or "no output").strip()[-800:]
        return EvalResult(float("-inf"), error=err or "no parseable output",
                          seconds=r.seconds)
    metrics = payload.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    if payload.get("error"):
        return EvalResult(float("-inf"), metrics=metrics,
                          error=str(payload["error"])[:800], seconds=r.seconds)
    try:
        return EvalResult(float(payload["score"]), metrics=metrics, seconds=r.seconds)
    except (KeyError, TypeError, ValueError):
        return EvalResult(float("-inf"), metrics=metrics,
                          error="malformed score payload", seconds=r.seconds)


class _StellarP2Task:
    name = "stellar_p2"
    wiki_dir = _DIR / "wiki"
    description = (_DIR / "description.md").read_text()
    noise: dict = {}         # deterministic sim: no tie re-run machinery
    extra_splits = ()

    def __init__(self):
        self._bcache: dict[str, dict] = {}   # code sha -> returned boundary
        self._arch_keys: set[str] | None = None
        self._arch_best = float("-inf")

    def seed_code(self) -> str:
        from tasks.stellar_p2.seed import seed_code
        return seed_code()

    # -- archive --------------------------------------------------------
    def _arch_load(self) -> None:
        if self._arch_keys is not None:
            return
        self._arch_keys = set()
        if _ARCHIVE.exists():
            for line in _ARCHIVE.read_text().splitlines():
                try:
                    e = json.loads(line)
                    self._arch_keys.add(e["key"])
                    self._arch_best = max(self._arch_best, e.get("shaped", -9e9))
                except (ValueError, KeyError):
                    continue

    def _archive(self, entries: list[dict], code_sha: str, fidelity: str) -> None:
        """Append noteworthy deduped boundaries to the global archive."""
        self._arch_load()
        _ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        with _ARCHIVE.open("a") as f:
            for e in entries:
                shaped = e.get("shaped", float("-inf"))
                if shaped < _ARCH_MIN and shaped <= self._arch_best:
                    continue
                key = _bkey(e["boundary"])
                if key in self._arch_keys:
                    continue
                self._arch_keys.add(key)
                self._arch_best = max(self._arch_best, shaped)
                f.write(json.dumps({
                    "ts": round(time.time(), 1), "key": key, "code_sha": code_sha,
                    "fidelity": fidelity, "shaped": shaped, "p2": e.get("p2"),
                    "feasibility": e.get("feasibility"),
                    "boundary": e["boundary"]}) + "\n")

    # -- evaluation -----------------------------------------------------
    def evaluate(self, code: str, split: str) -> EvalResult:
        if split == "train":
            return self._train(code)
        if split not in ("val", "public", "private"):
            return EvalResult(float("-inf"), error=f"unknown split: {split}")
        sha = _sha(code)
        if sha not in self._bcache:   # e.g. gate probing an unevaluated candidate
            r = self._train(code)
            if sha not in self._bcache:
                return EvalResult(float("-inf"),
                                  error=f"no boundary to verify: {r.error}")
        boundary = self._bcache[sha]
        res = verify_boundary(boundary, official=(split == "private"),
                              fidelity=_FIDELITY.get(split, "low_fidelity"))
        d = bank_distance(boundary)
        if not res.error and d is not None:
            res.metrics["bank_dist"] = round(d, 6)
        if not res.error and split in ("val", "private"):
            kind = "official" if split == "private" else _FIDELITY[split]
            self._archive([{"shaped": res.metrics.get("shaped_score", res.score),
                            "p2": res.metrics.get("p2_score"),
                            "feasibility": res.metrics.get("feasibility"),
                            **({"bank_dist": round(d, 6)} if d is not None else {}),
                            "boundary": boundary}], sha, kind)
        if split == "private":
            if not res.error:
                res.metrics["submittable"] = bool(
                    res.score > 0 and (d is None or d >= _NOVELTY_MIN))
            return res  # official truth, never shaped
        return _novelty_shape(res, d)

    def _train(self, code: str) -> EvalResult:
        cfg = {**_TRAIN, "fidelity": _FIDELITY["train"],
               "seed_bank": [{"boundary": e["boundary"],
                              "score": e["official_score"],
                              "feasibility": e["official_feasibility"]}
                             for e in _BANK]}
        script = _TEMPLATE.replace("__CODE__", repr(code)).replace(
            "__CFG__", repr(json.dumps(cfg)))
        timeout = cfg["cpu_budget"] + _SLACK
        r = _runner(script, timeout=timeout, cpus=cfg["workers"])
        if r is None:
            return EvalResult(float("-inf"), error="docker image unavailable "
                              f"(build {_IMAGE} from {_DOCKERFILE})")
        res = _parse(r, timeout)
        if res.error:
            return res
        m = res.metrics
        boundary = m.pop("boundary", None)
        arch = m.pop("archive", [])
        if not isinstance(boundary, dict):
            return EvalResult(float("-inf"), error="no boundary in eval payload",
                              seconds=res.seconds)
        sha = _sha(code)
        self._bcache[sha] = boundary
        d = bank_distance(boundary)
        if d is not None:
            m["bank_dist"] = round(d, 6)
        try:
            self._archive(arch + [{"shaped": res.score, "p2": m.get("p2_score"),
                                   "feasibility": m.get("feasibility"),
                                   **({"bank_dist": round(d, 6)} if d is not None else {}),
                                   "boundary": boundary}], sha, _FIDELITY["train"])
        except Exception as e:  # archive is bookkeeping: never fail an eval on it
            m["archive_error"] = str(e)[:200]
        return _novelty_shape(res, d)

    # -- frontend -------------------------------------------------------
    def render(self, code: str, result: EvalResult) -> dict:
        boundary = self._bcache.get(_sha(code)) \
            or (result.metrics or {}).get("boundary")
        if not isinstance(boundary, dict):
            return {"kind": "stellarator", "sections": []}
        rc = np.asarray(boundary["r_cos"], float)
        zs = np.asarray(boundary["z_sin"], float)
        nfp = int(boundary.get("n_field_periods", 1))
        ntor = (rc.shape[1] - 1) // 2
        mm = np.arange(rc.shape[0])[:, None]
        nn = np.arange(-ntor, ntor + 1)[None, :]
        theta = np.linspace(0, 2 * np.pi, 72)
        sections = []
        for phi in np.linspace(0, np.pi / nfp, 4):  # half a field period
            ang = mm[None] * theta[:, None, None] - nfp * nn[None] * phi
            sections.append({
                "phi_deg": round(float(np.degrees(phi)), 1),
                "r": np.round((rc[None] * np.cos(ang)).sum((1, 2)), 4).tolist(),
                "z": np.round((zs[None] * np.sin(ang)).sum((1, 2)), 4).tolist()})
        met = result.metrics or {}
        return {"kind": "stellarator", "sections": sections, "nfp": nfp,
                "p2_score": met.get("p2_score"), "feasibility": met.get("feasibility")}


TASK = _StellarP2Task()
