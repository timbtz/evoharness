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
# B6-nae-independent asks whether this harness can reach a feasible QI boundary
# UNAIDED. With the bank importable, a writer can always call fm.seed_bank(i) and
# quietly defeat the experiment, so the branch must hide it rather than ask
# nicely: STELLAR_NO_BANK=1 empties it for the whole process (2026-07-27).
if os.environ.get("STELLAR_NO_BANK") == "1":
    _BANK = []
_SLACK = 240.0   # container start + jax import/JIT + the last in-flight eval
# overshooting the CPU deadline + final re-score — high-mode (11,21) boundaries
# take 12-27s per forward call, so every tail item is ~20x the mp=1 case
_MEM_MB = 4096   # jax wants >= 2 GB; fork workers add ~0.5 GB each over COW

# Plan-2 margin tools in the train template (exact aspect value + gradient +
# margin walk, all eval-free). STELLAR_MARGIN_GRAD=0 = the A/B control arm:
# the fm methods refuse AND the description section is stripped, so the control
# writer is not told about a tool it does not have.
_GRAD_DOC = ("<!--MARGIN-GRAD-->", "<!--/MARGIN-GRAD-->")

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

def _honest(real, feas):
    """Score discounted to a low-tolerance margin (audit 2026-07-27). The official
    rule accepts any violation <= 1%, and P2 rises ~0.92 per unit of feasibility
    spent, so raw score rewards walking aspect ratio into the wall rather than
    finding better structure. This is what the search actually optimizes now;
    `p2_score` stays the raw official number."""
    if real <= 0:
        return real
    return real - CFG.get("margin_slope", 0.92) * max(
        0.0, feas - CFG.get("margin_target", 0.002))

def _bdict(b):
    return json.loads(b.model_dump_json())
'''

# ---- soft-fail grading, shared host/template (Plan 1, 2026-07-30) ----------
# exec'd host-side too so golden tests assert the exact in-container rule
_SOFT_SRC = '''
_SF_FLOOR = -1000.0  # every converged shaped score (= -feasibility, |feas| ~ O(1)
# in all 50k observed evals) strictly dominates every soft-fail sentinel
def _soft_penalty(fsqr, fsqz, fsql, ftol):
    """Graded fitness for a non-converged solve, in [-1001, -1000]: higher =
    closer to convergence (log10 of summed force residuals vs requested ftol)."""
    import math
    resid = max(float(fsqr) + float(fsqz) + float(fsql), 1e-300)
    g = min(max((math.log10(resid) - math.log10(float(ftol))) / 12.0, 0.0), 1.0)
    return _SF_FLOOR - g
'''
exec(_SOFT_SRC)  # noqa: S102 — defines _soft_penalty for the golden tests

# ---- eval-efficiency patch (Plan 1): hot restart + graded soft-fail --------
# Train-loop only: _VERIFY (val/public/private) never contains this block and
# the final authoritative re-score runs strict (pure cold semantics), so the
# anti-forging boundary is untouched. The restart cache is template-owned and
# per pool worker; candidate code cannot reach or seed it.
# Kill-switches: STELLAR_HOT_RESTART=0 / STELLAR_SOFT_FAIL=0 (host env) — with
# both off, run_vmec is never patched and the old path is restored.
_HOTPATCH = _SOFT_SRC + '''
import vmecpp as _vmecpp
from constellaration.mhd import vmec_utils as _vu

_HR_ON = bool(CFG.get("hot_restart"))
_SF_ON = bool(CFG.get("soft_fail"))
_HR_TOL = float(CFG.get("hot_restart_tol", 1e-3))
# "single_stage": near-parent children skip the vlf preset's ns=25 stage (a
# fixed ~2000 wasted iterations: ftol 1e-17 is unreachable) and solve cold at
# ns=71 — measured bit-identical metrics to the multigrid path. "restart"
# additionally passes restart_from=<parent> (true hot restart): measured to
# converge to a DIFFERENT equilibrium at vlf's loose ftol (iota0 up to ~2%,
# min L-gradB up to 30% => p2 drift up to 0.19) — kept only for experiments.
_HR_MODE = str(CFG.get("hot_restart_mode", "single_stage"))
_SF_NITER = int(CFG.get("soft_fail_niter", 5000))
_HR_CACHE = []   # per pool worker: (nfp, r_cos, z_sin, VmecOutput), newest last
_STRICT = False  # set by _worker for the final re-score: cold, hard-fail

class _SoftFail(Exception):
    def __init__(self, w, ftol):
        self.info = {"fsqr": w.fsqr, "fsqz": w.fsqz, "fsql": w.fsql,
                     "ftol": ftol, "niter": int(w.niter)}
        super().__init__("VMEC not converged after %d iters: fsqr=%.1e fsqz=%.1e"
                         " fsql=%.1e (ftol %.0e)"
                         % (w.niter, w.fsqr, w.fsqz, w.fsql, ftol))

def _hr_parent(nfp, rc, zs, indata):
    """Closest cached converged parent on an IDENTICAL grid within _HR_TOL
    (vlf resolution follows the boundary's nonzero modes, so grids can differ
    between parent and child; restart is only well-posed on the same grid)."""
    best = None
    for pnfp, prc, pzs, pout in _HR_CACHE:
        pin = pout.input
        if (pnfp != nfp or prc.shape != rc.shape or pin.mpol != indata.mpol
                or pin.ntor != indata.ntor or pin.ntheta != indata.ntheta
                or pin.nzeta != indata.nzeta
                or pin.ftol_array[-1] != indata.ftol_array[-1]):  # same fidelity
            continue
        d = max(np.abs(prc - rc).max(), np.abs(pzs - zs).max())
        if d <= _HR_TOL and (best is None or d < best[0]):
            best = (d, pout)
    return best[1] if best else None

def _run_vmec(boundary, mhd_parameters, vmec_settings):
    """Drop-in for vmec_utils.run_vmec: when a cached converged parent is close
    enough, solve the child in a single multigrid stage at the final ns
    (default mode skips the wasted ns=25 stage; "restart" mode also passes the
    parent state), and grade non-convergence instead of erasing it."""
    indata = _vu.build_vmecpp_indata(mhd_parameters=mhd_parameters,
                                     boundary=boundary,
                                     vmec_settings=vmec_settings)
    if _SF_ON and not _STRICT:
        # Surface non-convergence as data, and cap the final stage so it
        # surfaces INSIDE the 60s eval kill (vlf's niter=20000 would take
        # ~200s: today those evals die as information-free pool kills).
        # The cap only binds where the old path timed out anyway: worst
        # healthy eval observed is ~27s ~ 3000 iters.
        indata.return_outputs_even_if_not_converged = True
        indata.niter_array = indata.niter_array.copy()
        indata.niter_array[-1] = min(indata.niter_array[-1], _SF_NITER)
    rc = np.asarray(boundary.r_cos, float)
    zs = np.asarray(boundary.z_sin, float)
    nfp = int(boundary.n_field_periods)
    out, warm = None, False
    parent = None if (_STRICT or not _HR_ON) else _hr_parent(nfp, rc, zs, indata)
    if parent is not None:
        wi = indata.model_copy(deep=True)
        wi.return_outputs_even_if_not_converged = False  # warm miss -> cold retry
        wi.ns_array, wi.ftol_array, wi.niter_array = (
            wi.ns_array[-1:], wi.ftol_array[-1:], wi.niter_array[-1:])
        try:
            out = _vmecpp.run(wi, verbose=vmec_settings.verbose,
                              max_threads=vmec_settings.max_threads,
                              restart_from=(parent if _HR_MODE == "restart"
                                            else None))
            warm = True
        except Exception:
            out = None                                   # fall back to cold
    if out is None:
        out = _vmecpp.run(indata, verbose=vmec_settings.verbose,
                          max_threads=vmec_settings.max_threads)
    _run_vmec.last_warm = warm
    w = out.wout
    ftol = float(indata.ftol_array[-1])
    if max(w.fsqr, w.fsqz, w.fsql) > ftol:  # reachable only with _SF_ON
        raise _SoftFail(w, ftol)
    if _HR_ON and not _STRICT:
        _HR_CACHE.append((nfp, rc, zs, out))
        del _HR_CACHE[:-4]                  # keep the last 4 converged parents
    return _vu.vmecppwout_from_wout(w)

_run_vmec.last_warm = False
if _HR_ON or _SF_ON:
    _vu.run_vmec = _run_vmec  # forward_model resolves it via the module attr
'''

# ---- Plan-2 margin gradients: exact, eval-free aspect control -------------
# Numpy port of experiments/diffscore/margins_jax.aspect_ratio + its analytic
# gradient (JAX parity is a golden test: value 1e-13, gradient 1e-9 vs the
# autodiff reference). numpy and NOT jax on purpose — this runs in the parent
# process, which forks the eval pool; a jax op here would leave an initialized
# XLA runtime in the parent and the pool respawn after an eval timeout would
# fork it (the documented deadlock). Aspect is the only scored constraint that
# is exactly boundary-only (Plan-2 pre-flight), so it is the only one that can
# be handed to candidate code for free; qi/iota/mirror stay measured, and
# elongation needs the magnetic axis, i.e. an equilibrium.
# Kill-switch: STELLAR_MARGIN_GRAD=0 (also strips the description section, so
# the A/B control run neither has nor is told about the tool).
_GRAD_SRC = '''
_ASPECT_BOUND = 10.0            # problems.py:214 -> violation = (aspect-10)/10
_GRID = (128, 128)              # theta/phi samples; exact for band-limited R,Z
_ANG = {}                       # (mpol, cols, nfp) -> angle tables

def _ang_tables(rc, nfp):
    key = (rc.shape[0], rc.shape[1], nfp)
    if key not in _ANG:
        _ANG.clear()            # one boundary shape at a time: tables are ~30 MB
        mpol, cols = rc.shape
        ntor = (cols - 1) // 2
        m = np.arange(mpol, dtype=float)[:, None, None, None]
        n = np.arange(-ntor, ntor + 1, dtype=float)[None, :, None, None]
        nth, nph = _GRID
        th = np.linspace(0.0, 2.0 * np.pi, nth, endpoint=False)
        ph = np.linspace(0.0, 2.0 * np.pi / nfp, nph, endpoint=False)
        ang = m * th[None, None, :, None] - nfp * n * ph[None, None, None, :]
        _ANG[key] = (m[:, :, 0, 0], np.cos(ang), np.sin(ang),
                     2.0 * np.pi / nth, nph)
    return _ANG[key]

def _aspect_full(r_cos, z_sin, nfp):
    """(aspect, d aspect/d r_cos, d aspect/d z_sin) from the boundary alone —
    VMEC's definition: Aminor = sqrt(<area(phi)>/pi), Rmajor = V/(2 pi^2
    Aminor^2), aspect = Rmajor/Aminor. Trapezoid sums are exact here, so this
    reproduces wout.aspect to ~1e-13 without solving anything."""
    rc = np.asarray(r_cos, float)
    zs = np.asarray(z_sin, float)
    m, cos_a, sin_a, dth, nph = _ang_tables(rc, int(nfp))
    R = np.einsum('mn,mntp->tp', rc, cos_a)
    dRdt = np.einsum('mn,mntp->tp', -m * rc, sin_a)
    dZdt = np.einsum('mn,mntp->tp', m * zs, cos_a)
    Z = np.einsum('mn,mntp->tp', zs, sin_a)
    raw_area = -np.sum(Z * dRdt, axis=0) * dth        # signed area per phi
    sa = np.where(raw_area >= 0.0, 1.0, -1.0)
    A = float(np.mean(np.abs(raw_area)))
    raw_vol = float(np.mean(np.sum(0.5 * R**2 * dZdt, axis=0) * dth)) * 2.0 * np.pi
    sv, V = (1.0, raw_vol) if raw_vol >= 0.0 else (-1.0, -raw_vol)
    aspect = V / (2.0 * np.pi**2 * (A / np.pi) ** 1.5)
    # aspect is linear in V and ~A^-3/2, so the chain rule collapses to two terms
    w = dth / nph
    dA_rc = w * m * np.einsum('mntp,tp->mn', sin_a, Z * sa)
    dA_zs = -w * np.einsum('mntp,tp->mn', sin_a, dRdt * sa)
    c = sv * 2.0 * np.pi * w
    dV_rc = c * np.einsum('mntp,tp->mn', cos_a, R * dZdt)
    dV_zs = c * 0.5 * m * np.einsum('mntp,tp->mn', cos_a, R**2)
    return (aspect, aspect * (dV_rc / V - 1.5 * dA_rc / A),
            aspect * (dV_zs / V - 1.5 * dA_zs / A))

def _free_mask(rc, zs):
    """Coefficients a step may touch: stellarator symmetry pins r_cos[0,n<0]
    and z_sin[0,n<=0] at zero."""
    ntor = (rc.shape[1] - 1) // 2
    mr, mz = np.ones_like(rc), np.ones_like(zs)
    mr[0, :ntor] = 0.0
    mz[0, :ntor + 1] = 0.0
    return mr, mz

def _aspect_walk(r_cos, z_sin, nfp, target, cap):
    """Newton walk in boundary space onto a target aspect violation. Every
    iteration is exact and costs no eval, so this LANDS on the target instead
    of predicting it; `cap` bounds the total max-coefficient displacement."""
    rc = np.array(r_cos, float)
    zs = np.array(z_sin, float)
    mr, mz = _free_mask(rc, zs)
    moved = 0.0
    for _ in range(6):
        a, g_rc, g_zs = _aspect_full(rc, zs, nfp)
        want = float(target) - (a - _ASPECT_BOUND) / _ASPECT_BOUND
        room = cap - moved
        if abs(want) < 1e-9 or room <= 0.0:
            break
        g = np.concatenate([(g_rc * mr).ravel(),
                            (g_zs * mz).ravel()]) / _ASPECT_BOUND
        gg = float(g @ g)
        if gg <= 0.0:
            break
        x = g * (want / gg)                    # min-norm step, violation units
        big = float(np.max(np.abs(x)))
        if big > room:
            x *= room / big
            big = room
        moved += big
        rc = rc + x[:rc.size].reshape(rc.shape)
        zs = zs + x[rc.size:].reshape(zs.shape)
    return rc, zs
'''
exec(_GRAD_SRC)  # noqa: S102 — the golden tests check this exact source

# ---- optimizer-run template: candidate code + metered fm handle ------------
_TEMPLATE = _PRELUDE + _HOTPATCH + _GRAD_SRC + '''
def _worker(job):
    """One forward eval in a pool worker; never raises.
    job = (boundary, fid, strict) — strict is the final authoritative re-score:
    cold solve, hard failure, exactly the pre-Plan-1 semantics."""
    boundary, fid, strict = job
    global _STRICT
    _STRICT = strict
    _t0 = time.monotonic()
    try:
        b = _srf.SurfaceRZFourier.model_validate(boundary)
        m, _ = _fmod.forward_model(b, settings=_settings(fid))
    except _SoftFail as e:
        return None, e.info               # graded channel: dict, not message
    except Exception as e:
        return None, str(e)[:300]
    finally:
        _STRICT = False
    d = {k: v for k, v in m.model_dump().items() if isinstance(v, (int, float))}
    d["feasibility"], d["p2_score"], d["shaped_score"] = _assess(m)
    d["honest_score"] = _honest(d["p2_score"], d["feasibility"])
    d["fidelity"] = fid or CFG["fidelity"]
    if _HR_ON or _SF_ON:
        d["warm_start"] = _run_vmec.last_warm
        d["eval_seconds"] = round(time.monotonic() - _t0, 2)
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

_GRACE = 60.0  # a SEARCH batch may overrun cpu_budget by this much, no more
# The container's own hard deadline: the host kills it at cpu_budget + slack, so
# stop a little before that and leave the remainder for pool teardown + emitting
# the JSON payload. The final re-score is mandatory, unmetered work that by
# construction starts AFTER the search has spent cpu_budget — clamping it to the
# search deadline floors it at 5s and -infs every candidate that uses its full
# budget (first candidate of campaign 2, 2026-07-27). The slack is exactly what
# it must be allowed to spend.
_HARD = CFG["cpu_budget"] + CFG.get("slack", 240.0) - 45.0

def _pool_eval(batch, fidelity=None, final=False):
    global _POOL
    limit = CFG["eval_timeout"] * _math.ceil(len(batch) / CFG["workers"]) + 15.0
    if fidelity == "low_fidelity":
        limit *= 2  # tighter force tolerance runs longer
    # Clamp to the wall-clock actually left. Without this a batch submitted near
    # the deadline could legally run eval_timeout*ceil(n/workers) seconds (15
    # boundaries / 2 workers / 180s = 24 min), blowing past the host timeout of
    # cpu_budget+_SLACK: the candidate is killed, returns -inf, and its whole
    # slot plus ~12 min of wall clock is lost. 37 of 456 campaign candidates died
    # exactly this way, 30% of all eval wall-clock (audit 2026-07-27).
    deadline = _HARD if final else CFG["cpu_budget"] + _GRACE
    limit = min(limit, max(deadline - (time.monotonic() - _T0), 30.0 if final else 5.0))
    try:
        return _POOL.map_async(_worker,
                               [(b, fidelity, final) for b in batch]).get(limit)
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
        self.warmed, self.soft = 0, 0   # hot-restart / soft-fail eval counters
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
            if ok is None and isinstance(err, dict):
                # non-converged solve: graded sentinel strictly below every
                # converged score (never enters _LOG/archive; train-only signal)
                self.soft += 1
                self.last_error = ("VMEC not converged (fsqr=%.1e fsqz=%.1e "
                                   "fsql=%.1e)" % (err["fsqr"], err["fsqz"],
                                                   err["fsql"]))
                s = _soft_penalty(err["fsqr"], err["fsqz"], err["fsql"], err["ftol"])
                out.append({"soft_fail": True, "shaped_score": s,
                            "honest_score": s, "p2_score": 0.0,
                            "feasibility": float("inf"),
                            "fidelity": fidelity or CFG["fidelity"], **err})
            elif ok is None:
                self.last_error = err
                out.append(None)
            else:
                d, bd = ok
                self.warmed += int(bool(d.get("warm_start")))
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
        # An empty bank means STELLAR_NO_BANK=1: this run is the unaided-discovery
        # test. Say so loudly — a bare IndexError just costs the writer a candidate.
        if not CFG["seed_bank"]:
            raise RuntimeError(
                "seed bank is DISABLED for this run (independent-basin test): "
                "no public submissions are available. Build the boundary from "
                "fm.seed_nae(...) / fm.seed_ellipse(...) only.")
        return json.loads(json.dumps(CFG["seed_bank"][int(i)]["boundary"]))
    # ---- exact boundary-margin tools (Plan 2): no eval budget, no solver ----
    def aspect(self, boundary):
        """Exact VMEC aspect ratio of a boundary (free). Same number the
        simulator would report, computed from the Fourier coefficients."""
        self._grad_on()
        a, _, _ = _aspect_full(boundary["r_cos"], boundary["z_sin"],
                               boundary["n_field_periods"])
        return a
    def margin_grad(self, boundary):
        """Exact aspect-ratio margin and its gradient (free). Returns
        {"aspect", "violation", "grad_r_cos", "grad_z_sin"} where the gradients
        are d(normalized violation)/d(coefficient), already zeroed on the
        coefficients stellarator symmetry pins."""
        self._grad_on()
        a, g_rc, g_zs = _aspect_full(boundary["r_cos"], boundary["z_sin"],
                                     boundary["n_field_periods"])
        mr, mz = _free_mask(np.asarray(boundary["r_cos"], float),
                            np.asarray(boundary["z_sin"], float))
        return {"aspect": a, "violation": (a - _ASPECT_BOUND) / _ASPECT_BOUND,
                "grad_r_cos": (g_rc * mr / _ASPECT_BOUND).tolist(),
                "grad_z_sin": (g_zs * mz / _ASPECT_BOUND).tolist()}
    def margin_step(self, boundary, target=0.002, cap=3e-3):
        """Boundary moved so its aspect violation equals `target` (free): a
        min-norm Newton walk on the exact gradient, capped at `cap` in
        max-coefficient distance. The geometric margin lands where you asked;
        the equilibrium-carried margins (qi above all) move too and are NOT
        modelled here — re-evaluate before you trust the result."""
        self._grad_on()
        rc, zs = _aspect_walk(boundary["r_cos"], boundary["z_sin"],
                              boundary["n_field_periods"], target, cap)
        b = json.loads(json.dumps(boundary))
        b["r_cos"], b["z_sin"] = rc.tolist(), zs.tolist()
        return b
    def _grad_on(self):
        if not CFG.get("margin_grad"):
            raise RuntimeError("margin-gradient tools are disabled for this run")
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
    # (same hang protection); candidates cannot forge this number. Retried: a
    # transient hard-timeout under host CPU contention must not -inf a full run
    # whose returned boundary is fine (cost run s103-92006336, 2026-07-25)
    for _retry in range(3):
        _ok, _err = _pool_eval([_boundary], final=True)[0]
        if _ok is not None:
            break
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
        **({"evals_warm": _fm.warmed, "evals_soft": _fm.soft}
           if _HR_ON or _SF_ON else {}),
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


_EXPORT_MIN = 1e-3               # submit_export's hard refusal bar (never shaped)
# Train/val novelty ramp. The audit found max-coefficient distance is a WEAK
# novelty measure: the 0.6400 champion sat at bank_dist 2.6e-3 — outside the
# export ball — while being cosine 0.999989 to davidkh's public #1 boundary,
# i.e. the same shape. Widening the ramp past the export bar keeps the gradient
# pointing out of the public basin instead of switching off the moment a
# candidate is barely exportable. Per-campaign knob, default = the old behaviour:
#   STELLAR_NOVELTY='{"min":0.003,"pen":0.05}'
_NOVELTY = {"min": 1e-3, "pen": 0.05}
_NOVELTY.update(json.loads(os.environ.get("STELLAR_NOVELTY", "{}")))
_NOVELTY_MIN, _NOVELTY_PEN = float(_NOVELTY["min"]), float(_NOVELTY["pen"])


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


def bank_kin(boundary: dict) -> tuple[float | None, float | None]:
    """(max-coefficient distance, cosine) to the closest same-nfp public seed-bank
    boundary, scale-normalized and padded to a common canvas. The distance is the
    export guard's exact metric; the cosine is what that distance hides — a
    boundary can clear the 1e-3 ball and still be cosine 0.999989 to a public
    submission, i.e. the same shape rescaled (audit 2026-07-27)."""
    if not _BANK:
        return None, None

    def pad_to(a, shape):
        out = np.zeros(shape)
        r, c = a.shape
        off = (shape[1] - c) // 2
        out[:r, off:off + c] = a
        return out

    rc0, zs0 = _scale_norm(boundary)
    best, cos = None, None
    for e in _BANK:
        bb = e["boundary"]
        if bb.get("n_field_periods") != boundary.get("n_field_periods"):
            continue
        rc1, zs1 = _scale_norm(bb)
        shape = (max(rc0.shape[0], rc1.shape[0]), max(rc0.shape[1], rc1.shape[1]))
        a0, b0 = pad_to(rc0, shape), pad_to(rc1, shape)
        a1, b1 = pad_to(zs0, shape), pad_to(zs1, shape)
        d = max(np.abs(a0 - b0).max(), np.abs(a1 - b1).max())
        if best is None or d < best:
            x = np.concatenate([a0.ravel(), a1.ravel()])
            y = np.concatenate([b0.ravel(), b1.ravel()])
            n = float(np.linalg.norm(x) * np.linalg.norm(y))
            best, cos = d, (round(float(x @ y / n), 6) if n else None)
    return best, cos


def bank_distance(boundary: dict) -> float | None:
    """Distance half of bank_kin — submit_export's guard imports this name."""
    return bank_kin(boundary)[0]


# Feasibility-margin shaping (audit 2026-07-27). The official rule accepts any
# normalized violation <= 1% and pays ~0.92 score per unit of it, so raw score
# rewards camping the aspect-ratio wall. Train/val fitness is therefore the score
# discounted to _MARGIN_TARGET; private stays the raw official truth so the
# leaderboard number is never distorted. Tune per run without a code change:
# STELLAR_MARGIN='{"target":0.002,"slope":0.92}'
_MARGIN = {"target": 0.002, "slope": 0.92}
_MARGIN.update(json.loads(os.environ.get("STELLAR_MARGIN", "{}")))


def _margin_shape(res: EvalResult) -> EvalResult:
    """Charge the candidate for the feasibility tolerance it spends."""
    m = res.metrics or {}
    p2, feas = m.get("p2_score"), m.get("feasibility")
    if res.error or p2 is None or feas is None or p2 <= 0:
        return res
    pen = round(_MARGIN["slope"] * max(0.0, feas - _MARGIN["target"]), 6)
    m["honest_score"] = round(p2 - pen, 6)
    m["margin_penalty"] = pen
    if not pen:
        return res
    return EvalResult(res.score - pen, metrics=m, error=res.error,
                      seconds=res.seconds)


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


def _description() -> str:
    """Writer-visible task description. With the margin tools switched off the
    section documenting them is stripped, so the A/B control run is neither
    given the tool nor told it exists."""
    text = (_DIR / "description.md").read_text()
    if os.environ.get("STELLAR_MARGIN_GRAD", "1") != "0":
        return text
    head, _, rest = text.partition(_GRAD_DOC[0])
    return head + rest.partition(_GRAD_DOC[1])[2]


class _StellarP2Task:
    name = "stellar_p2"
    wiki_dir = _DIR / "wiki"
    description = _description()
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
        d, cos = bank_kin(boundary)
        if not res.error and d is not None:
            res.metrics["bank_dist"] = round(d, 6)
            if cos is not None:
                res.metrics["bank_cos"] = cos
        if not res.error and split in ("val", "private"):
            kind = "official" if split == "private" else _FIDELITY[split]
            self._archive([{"shaped": res.metrics.get("shaped_score", res.score),
                            "p2": res.metrics.get("p2_score"),
                            "feasibility": res.metrics.get("feasibility"),
                            **({"bank_dist": round(d, 6)} if d is not None else {}),
                            "boundary": boundary}], sha, kind)
        if split == "private":
            if not res.error:
                res.metrics["submittable"] = bool(   # export guard's bar, not the
                    res.score > 0 and (d is None or d >= _EXPORT_MIN))  # ramp's
                p2, fe = res.metrics.get("p2_score"), res.metrics.get("feasibility")
                if p2 and fe is not None:  # reported, never subtracted from truth
                    res.metrics["honest_score"] = round(
                        p2 - _MARGIN["slope"] * max(0.0, fe - _MARGIN["target"]), 6)
                    res.metrics["tolerance_used"] = round(fe / 1e-2, 4)
            return res  # official truth, never shaped
        return _margin_shape(_novelty_shape(res, d))

    def _train(self, code: str) -> EvalResult:
        cfg = {**_TRAIN, "fidelity": _FIDELITY["train"],
               "margin_target": _MARGIN["target"],
               "margin_slope": _MARGIN["slope"],
               "slack": _SLACK,   # container's own deadline = cpu_budget + this
               # Plan 1 eval-efficiency knobs (read per call so kill-switches
               # work without a reimport): STELLAR_HOT_RESTART=0 / _SOFT_FAIL=0
               # restore the pre-patch solve path byte-identically
               "hot_restart": os.environ.get("STELLAR_HOT_RESTART", "1") != "0",
               "soft_fail": os.environ.get("STELLAR_SOFT_FAIL", "1") != "0",
               "hot_restart_mode": os.environ.get(
                   "STELLAR_HOT_RESTART_MODE", "single_stage"),
               "hot_restart_tol": float(os.environ.get(
                   "STELLAR_HOT_RESTART_TOL", "1e-3")),
               "soft_fail_niter": int(os.environ.get(
                   "STELLAR_SOFT_FAIL_NITER", "5000")),
               # Plan-2 exact aspect gradient handed to candidate code (the
               # description section is gated at import; this flag per call)
               "margin_grad": os.environ.get("STELLAR_MARGIN_GRAD", "1") != "0",

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
        d, cos = bank_kin(boundary)
        if d is not None:
            m["bank_dist"] = round(d, 6)
            if cos is not None:
                m["bank_cos"] = cos
        try:
            self._archive(arch + [{"shaped": res.score, "p2": m.get("p2_score"),
                                   "feasibility": m.get("feasibility"),
                                   **({"bank_dist": round(d, 6)} if d is not None else {}),
                                   "boundary": boundary}], sha, _FIDELITY["train"])
        except Exception as e:  # archive is bookkeeping: never fail an eval on it
            m["archive_error"] = str(e)[:200]
        return _margin_shape(_novelty_shape(res, d))

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
