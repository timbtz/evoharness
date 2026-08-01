"""Offline validation for Plan 1 (hot restart + soft-fail), acceptance A(i)/B.

Samples parent/child boundary pairs from the stellar_p2 archive at max-coeff
distance 1e-4..1e-3, then — inside the pinned eval image, exercising the exact
_HOTPATCH string the train template ships — times cold vs warm child solves and
measures metric drift. Part 2 re-evals p2=0 near-miss boundaries (and one
1e-3-perturbed child each) through the soft-fail channel.

Usage: .venv/bin/python experiments/hot_restart_bench.py --pairs 55 --nearmiss 30
Raw per-pair JSONL goes to --out (default: runs/hot_restart_bench.jsonl).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from core.sandbox import docker_image_ready, run_python_docker  # noqa: E402
from tasks.stellar_p2 import task as st  # noqa: E402


def _mats(b):
    return np.asarray(b["r_cos"], float), np.asarray(b["z_sin"], float)


def _dist(a, b):
    ra, za = _mats(a)
    rb, zb = _mats(b)
    if ra.shape != rb.shape or a.get("n_field_periods") != b.get("n_field_periods"):
        return None
    return float(max(np.abs(ra - rb).max(), np.abs(za - zb).max()))


def sample_pairs(rows, n, lo=1e-4, hi=1e-3):
    """Consecutive archived boundaries of the same candidate = optimizer steps."""
    by_sha: dict[str, list] = {}
    for r in rows:
        by_sha.setdefault(r.get("code_sha", "?"), []).append(r)
    pairs = []
    for group in by_sha.values():
        for a, b in zip(group, group[1:]):
            d = _dist(a["boundary"], b["boundary"])
            if d is not None and lo <= d <= hi:
                pairs.append({"d": d, "parent": a["boundary"], "child": b["boundary"],
                              "parent_p2": a.get("p2"), "child_p2": b.get("p2")})
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    return pairs[:n]


_BENCH = '''import json, time, math
import numpy as np
DATA = json.loads(__DATA__)
CFG = {"hot_restart": True, "soft_fail": True,
       "hot_restart_tol": 1e-3, "soft_fail_niter": 5000}
__HOTPATCH__
from constellaration import forward_model as _fmod, problems as _problems
from constellaration.mhd import vmec_settings as _vs
from constellaration.geometry import surface_rz_fourier as _srf
_P2 = _problems.SimpleToBuildQIStellarator()
_S = _fmod.ConstellarationSettings(
    vmec_preset_settings=_vs.VmecPresetSettings(fidelity="very_low_fidelity"),
    turbulent_settings=None)

def fm(bd):
    t0 = time.time()
    try:
        m, _ = _fmod.forward_model(_srf.SurfaceRZFourier.model_validate(bd), settings=_S)
    except _SoftFail as e:
        return {"soft": e.info, "warm": _run_vmec.last_warm, "t": time.time() - t0}
    except Exception as e:
        return {"err": str(e)[:200], "t": time.time() - t0}
    feas = _P2.compute_feasibility(m)
    real = _P2._score(m) if _P2.is_feasible(m) else 0.0
    d = {k: v for k, v in m.model_dump().items() if isinstance(v, (int, float))}
    d.update(feasibility=feas, p2=real, shaped=(real if feas <= 1e-2 else -feas))
    return {"m": d, "warm": _run_vmec.last_warm, "t": time.time() - t0}

G = globals()
for i, p in enumerate(DATA["pairs"]):
    _HR_CACHE.clear()
    G["_HR_ON"] = True
    rp = fm(p["parent"])            # cold parent -> populates the warm-gate cache
    snap = _HR_CACHE[:]             # parent-only cache snapshot
    G["_HR_ON"] = False
    rc = fm(p["child"])             # cold child (old path)
    G["_HR_ON"] = True
    G["_HR_MODE"] = "single_stage"
    rss = fm(p["child"])            # warm: skip ns=25 stage, no restart state
    _HR_CACHE[:] = snap             # drop the child the ss run just cached
    G["_HR_MODE"] = "restart"
    rrs = fm(p["child"])            # warm: true hot restart off the parent
    _HR_CACHE[:] = snap
    print(json.dumps({"kind": "pair", "i": i, "d": p["d"],
                      "parent": rp, "cold": rc, "ss": rss, "rs": rrs}), flush=True)
G["_HR_ON"] = False                  # part 2: soft-fail channel only
rng = np.random.default_rng(0)
for i, bd in enumerate(DATA["nearmiss"]):
    r0 = fm(bd)
    rc0, zs0 = np.asarray(bd["r_cos"], float), np.asarray(bd["z_sin"], float)
    kid = dict(bd)
    kid["r_cos"] = (rc0 + 1e-3 * rng.standard_normal(rc0.shape) * (np.abs(rc0) > 1e-12)).tolist()
    kid["z_sin"] = (zs0 + 1e-3 * rng.standard_normal(zs0.shape) * (np.abs(zs0) > 1e-12)).tolist()
    r1 = fm(kid)
    print(json.dumps({"kind": "nearmiss", "i": i, "self": r0, "kid": r1}), flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=55)
    ap.add_argument("--nearmiss", type=int, default=30)
    ap.add_argument("--out", default=str(_ROOT / "runs" / "hot_restart_bench.jsonl"))
    a = ap.parse_args()

    rows = [json.loads(x) for x in st._ARCHIVE.read_text().splitlines()]
    pairs = sample_pairs(rows, a.pairs)
    near = [r["boundary"] for r in rows if (r.get("p2") or 0) == 0][: a.nearmiss]
    print(f"{len(pairs)} pairs, {len(near)} near-miss boundaries", flush=True)
    assert docker_image_ready(st._IMAGE, st._DOCKERFILE, str(st._DIR))

    script = _BENCH.replace("__HOTPATCH__", st._HOTPATCH).replace(
        "__DATA__", repr(json.dumps({"pairs": pairs, "nearmiss": near})))
    budget = 100.0 * len(pairs) + 75.0 * len(near) + 120.0
    t0 = time.time()
    r = run_python_docker(script, timeout=budget, mem_mb=4096, image=st._IMAGE, cpus=1)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"container: {time.time() - t0:.0f}s rc={r.returncode} "
          f"timed_out={r.timed_out} rows={len(lines)} -> {a.out}", flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])

    recs = [json.loads(ln) for ln in lines]
    for mode in ("ss", "rs"):
        ok = [x for x in recs if x["kind"] == "pair" and "m" in x.get("cold", {})
              and "m" in x.get(mode, {}) and x[mode]["warm"]]
        if not ok:
            continue
        ct = np.array([x["cold"]["t"] for x in ok])
        wt = np.array([x[mode]["t"] for x in ok])
        print(f"[{mode}] warm path taken on {len(ok)} pairs; "
              f"cold median {np.median(ct):.2f}s warm median {np.median(wt):.2f}s "
              f"ratio med {np.median(wt / ct):.2f} "
              f"(p10 {np.percentile(wt / ct, 10):.2f}, "
              f"p90 {np.percentile(wt / ct, 90):.2f})")
        for k in ("shaped", "p2", "feasibility",
                  "minimum_normalized_magnetic_gradient_scale_length"):
            v = np.array([abs(x[mode]["m"][k] - x["cold"]["m"][k]) for x in ok])
            print(f"  drift {k}: median {np.median(v):.2e} "
                  f"p90 {np.percentile(v, 90):.2e} max {v.max():.2e}")


if __name__ == "__main__":
    main()
