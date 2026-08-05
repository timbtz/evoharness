"""Version-drift razor: does a vmecpp 0.7.1 sidecar see the same boundary the
pinned 0.4.11 evaluator scores?

Plan 3's pre-flight calls this "independently valuable — do it even if the rest
stalls", and it is the gate on the whole sidecar architecture. A gradient
sidecar proposes boundaries; the pinned evaluator disposes. If the two solvers
disagree by more than the 1e-2 feasibility tolerance can absorb, the sidecar is
pointing at a different surface than the one we are scored on, and no amount of
adjoint accuracy fixes that.

Three stages, because the two solvers CANNOT live in one environment
(constellaration 0.2.6 hard-pins vmecpp==0.4.11):

  dump    (pinned image)  the exact VmecInput constellaration builds at vlf,
                          serialized to JSON — so the sidecar solves the SAME
                          problem, not a re-derived approximation. This is the
                          whole trick: identical input, one variable changed.
  solve   (sidecar venv)  vmecpp 0.7.1 runs those JSONs, dumps wout arrays.
  compare (pinned image)  the SAME metric code (Plan-2's JAX ports, validated
                          to 1e-13 against the pinned oracle) is applied to both
                          array sets. Metric implementation is therefore held
                          fixed; only the solver differs.

QI is out of scope here: it needs booz-xform, which is sdist-only and would mean
a cmake/netcdf build inside the sidecar. Covered: the P2 objective itself
(min normalized L-gradB) and 3 of the 5 constraints (aspect, edge iota, edge
mirror) plus elongation from wout inputs — enough to answer the architectural
question.

Usage (see run_version_drift.sh, which wires the three stages together):
  python experiments/version_drift.py --stage dump    --n 10     # in-image
  python experiments/version_drift.py --stage solve              # sidecar venv
  python experiments/version_drift.py --stage compare            # in-image
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "runs" / "polish" / "drift"
CACHE = _ROOT / "runs" / "diffscore" / "oracle_cache"

# wout fields the metric code needs, in constellaration's naming
FIELDS = ["rmnc", "zmns", "xm", "xn", "xm_nyq", "xn_nyq", "raxis_cc",
          "zaxis_cs", "iotaf", "bmnc", "gmnc", "bsupumnc", "bsupvmnc",
          "aspect", "Aminor_p", "volume_p",
          "normalized_toroidal_flux_full_grid_mesh",
          "normalized_toroidal_flux_half_grid_mesh"]

# vmecpp's native wout naming -> constellaration's. Verified against a live
# 0.7.1 solve: Aminor_p / volume_p / aspect / ns are direct name matches and
# come out BIT-IDENTICAL to the pinned side, so no aliasing is needed for the
# quantities the metrics consume. The two flux meshes do NOT exist in 0.7.1's
# VmecWOut, and phips/phipf are a DIFFERENT quantity (dflux/ds, not the s
# grid) — aliasing to them silently produced NaN. They are pure grid
# definitions, so _flux_meshes() derives them from ns for BOTH sides instead,
# checked against the pinned stored arrays.
ALIAS: dict = {}


def _flux_meshes(ns: int):
    """(full, half) normalized-toroidal-flux meshes, exactly as VMEC defines
    them: full = linspace(0,1,ns), half = full - 0.5/(ns-1)."""
    full = np.linspace(0.0, 1.0, int(ns))
    return full, full - 0.5 / (int(ns) - 1)


# ---------------------------------------------------------------------------
# stage 1: dump the pinned VmecInput (runs in the pinned image)
# ---------------------------------------------------------------------------

def stage_dump(n: int, fidelity: str) -> None:
    from constellaration.geometry import surface_rz_fourier as srf
    from constellaration.mhd import (ideal_mhd_parameters as imp,
                                     vmec_settings as vs, vmec_utils as vu)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (CACHE / "index.jsonl").read_text().splitlines()
            if x.strip()]
    rows = [r for r in rows if "boundary" in r and "error" not in r]
    # spread across strata so the answer is not about one shape family
    by_stratum: dict = {}
    for r in rows:
        by_stratum.setdefault(r.get("stratum", "?"), []).append(r)
    picked, i = [], 0
    while len(picked) < n and any(by_stratum.values()):
        keys = sorted(by_stratum)
        s = keys[i % len(keys)]
        if by_stratum[s]:
            picked.append(by_stratum[s].pop(0))
        i += 1

    index = []
    for r in picked:
        b = srf.SurfaceRZFourier.model_validate(r["boundary"])
        settings = vs.create_vmec_settings_from_preset(
            boundary=b, settings=vs.VmecPresetSettings(fidelity=fidelity))
        indata = vu.build_vmecpp_indata(
            mhd_parameters=imp.boundary_to_ideal_mhd_parameters(b),
            boundary=b, vmec_settings=settings)
        (OUT / f"{r['key']}.indata.json").write_text(indata.model_dump_json())
        index.append({"key": r["key"], "stratum": r.get("stratum"),
                      "npz_pinned": str(CACHE / f"{r['key']}.npz")})
        print(f"dumped {r['key']} ({r.get('stratum')})", flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"STAGE_DUMP_OK {len(index)} boundaries", flush=True)


# ---------------------------------------------------------------------------
# stage 2: solve with the 0.7.1 sidecar (runs in the sidecar venv)
# ---------------------------------------------------------------------------

def stage_solve() -> None:
    import vmecpp

    index = json.loads((OUT / "index.json").read_text())
    for row in index:
        key = row["key"]
        dst = OUT / f"{key}.new.npz"
        if dst.exists():
            print(f"skip {key} (already solved)", flush=True)
            continue
        raw = json.loads((OUT / f"{key}.indata.json").read_text())
        t0 = time.time()
        try:
            inp = vmecpp.VmecInput.model_validate(raw)
        except Exception as e:
            # 0.4.11 -> 0.7.1 schema drift is itself a finding, not a crash
            unknown = str(e)[:400]
            print(f"SCHEMA {key}: {unknown}", flush=True)
            row["error"] = f"schema: {unknown}"
            continue
        try:
            out = vmecpp.run(inp, max_threads=1, verbose=False)
        except Exception as e:
            print(f"SOLVE-FAIL {key}: {type(e).__name__}: {e}", flush=True)
            row["error"] = f"{type(e).__name__}: {e}"
            continue
        w = out.wout
        arrays = {}
        for f in FIELDS:
            for cand in ALIAS.get(f, [f]):
                if hasattr(w, cand):
                    arrays[f] = np.asarray(getattr(w, cand))
                    break
        missing = [f for f in FIELDS if f not in arrays]
        np.savez(dst, **arrays)
        row.update(solve_s=round(time.time() - t0, 1), missing=missing)
        print(f"solved {key} in {row['solve_s']}s "
              f"(missing {missing or 'nothing'})", flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print("STAGE_SOLVE_OK", flush=True)


# ---------------------------------------------------------------------------
# stage 3: same metric code on both array sets (runs in the pinned image)
# ---------------------------------------------------------------------------

def _metrics(z: dict) -> dict:
    from experiments.diffscore import lgradb_jax as lg, margins_jax as mj

    nfp = int(z["nfp"]) if "nfp" in z else int(z["_nfp"])
    mpol, ntor = int(z["mpol"]), int(z["ntor"])
    ns = int(z["ns"]) if "ns" in z else int(np.asarray(z["rmnc"]).shape[-1])
    s_full, s_half = _flux_meshes(ns)
    out = {
        "aspect_wout": float(z["aspect"]),
        "edge_iota_over_nfp": float(mj.edge_iota_over_nfp(z["iotaf"], nfp)),
        "edge_mirror": float(mj.edge_mirror_ratio(
            z["bmnc"], s_half, s_full,
            z["xm_nyq"], z["xn_nyq"], mpol, ntor, nfp)),
        # max_elongation_wout returns (max, per-plane array)
        "max_elongation": float(mj.max_elongation_wout(
            z["rmnc"], z["zmns"], z["xm"], z["xn"], z["raxis_cc"],
            z["zaxis_cs"], mpol, ntor, nfp)[0]),
        "min_L_gradB": float(lg.min_normalized_l_grad_b(
            z["rmnc"], z["zmns"], z["gmnc"], z["bmnc"], z["bsupumnc"],
            z["bsupvmnc"], z["xm"], z["xn"], z["xm_nyq"], z["xn_nyq"],
            ns, float(z["Aminor_p"]), mpol, ntor, nfp,
            bool(z["lasym"]) if "lasym" in z else False)),
    }
    out["p2_score"] = float(lg.p2_score(out["min_L_gradB"]))
    return out


def stage_compare() -> None:
    index = json.loads((OUT / "index.json").read_text())
    table = []
    for row in index:
        key = row["key"]
        new_p = OUT / f"{key}.new.npz"
        if not new_p.exists():
            table.append({"key": key, "error": row.get("error", "not solved")})
            continue
        old = dict(np.load(row["npz_pinned"], allow_pickle=True))
        new = dict(np.load(new_p, allow_pickle=True))
        # the derived meshes must reproduce what the PINNED solver stored;
        # the sidecar npz has no such field to check against (0.7.1 dropped it)
        f_ref, h_ref = _flux_meshes(int(old["ns"]))
        for nm, ref in (("normalized_toroidal_flux_full_grid_mesh", f_ref),
                        ("normalized_toroidal_flux_half_grid_mesh", h_ref)):
            err = float(np.abs(np.asarray(old[nm]) - ref).max())
            if err > 1e-12:
                raise SystemExit(f"{key}: {nm} reconstruction off by {err:g} "
                                 "— the derived grid is wrong, fix before use")
        # identical VmecInput => these are inputs, not outputs
        for k in ("nfp", "mpol", "ntor", "ns", "lasym"):
            if k in old:
                new.setdefault(k, old[k])
        try:
            mo, mn = _metrics(old), _metrics(new)
        except Exception as e:
            table.append({"key": key, "error": f"metrics: {type(e).__name__}: {e}"})
            continue
        drift = {k: float(mn[k] - mo[k]) for k in mo}
        table.append({"key": key, "stratum": row.get("stratum"),
                      "pinned": mo, "sidecar": mn, "drift": drift,
                      "solve_s": row.get("solve_s")})
        print(f"{key} ({row.get('stratum')}): "
              + " ".join(f"d{k}={v:+.3e}" for k, v in drift.items()), flush=True)

    ok = [t for t in table if "drift" in t]
    summary = {"n": len(table), "compared": len(ok),
               "failed": [t for t in table if "drift" not in t]}
    if ok:
        for k in ok[0]["drift"]:
            d = np.array([abs(t["drift"][k]) for t in ok])
            summary[k] = {"max_abs": float(d.max()),
                          "median_abs": float(np.median(d))}
    (OUT / "drift_report.json").write_text(
        json.dumps({"summary": summary, "rows": table}, indent=2))
    print("DRIFT_SUMMARY " + json.dumps(summary), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["dump", "solve", "compare",
                             "dirs-dump", "dirs-solve", "dirs-compare", "dirs-compare2"])
    ap.add_argument("--side", choices=["pinned", "sidecar"], default="pinned")
    ap.add_argument("--h", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--fidelity", default="very_low_fidelity")
    a = ap.parse_args()
    {"dump": lambda: stage_dump(a.n, a.fidelity),
     "solve": stage_solve, "compare": stage_compare,
     "dirs-dump": lambda: stage_dirs_dump(a.n, a.h, a.fidelity, a.seed),
     "dirs-solve": lambda: stage_dirs_solve(a.side),
     "dirs-compare": stage_dirs_compare,
     "dirs-compare2": stage_dirs_compare2}[a.stage]()
    return 0




# ---------------------------------------------------------------------------
# The sharper question the drift table raises
# ---------------------------------------------------------------------------
# A 0.006-0.010 P2-score offset between solvers would be survivable if the
# sidecar were only ever used for DIRECTIONS (Plan 3's "sidecar proposes,
# pinned disposes"). That is only true if the two solvers agree on the
# derivative, not just disagree by a constant. These stages measure it: the
# same boundary, the same random direction, a central difference of the P2
# objective computed by BOTH solvers from the SAME VmecInput JSON.
#
# Both sides call vmecpp.run() directly rather than the forward model, so the
# solver version is the only thing that differs — 0.4.11 in the pinned image,
# 0.7.1 in the sidecar venv.

# VD_DIRS lets a second step size run in its own directory, so a step-size
# control does not reuse the stencil solves of the first one.
DIRS = Path(os.environ.get("VD_DIRS", str(_ROOT / "runs" / "polish" / "dirs")))


def _free_mask(rc, zs):
    ntor = (rc.shape[1] - 1) // 2
    mr, mz = np.ones_like(rc), np.ones_like(zs)
    mr[0, :ntor] = 0.0
    mz[0, :ntor + 1] = 0.0
    return mr, mz


def stage_dirs_dump(n: int, h: float, fidelity: str, seed: int) -> None:
    from constellaration.geometry import surface_rz_fourier as srf
    from constellaration.mhd import (ideal_mhd_parameters as imp,
                                     vmec_settings as vs, vmec_utils as vu)

    DIRS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (CACHE / "index.jsonl").read_text().splitlines()
            if x.strip()]
    rows = [r for r in rows if "boundary" in r and "error" not in r
            and r.get("stratum") in ("feasible", "camped", "near")][:n]
    rng = np.random.default_rng(seed)

    index = []
    for r in rows:
        b = r["boundary"]
        rc, zs = np.asarray(b["r_cos"], float), np.asarray(b["z_sin"], float)
        mr, mz = _free_mask(rc, zs)
        # perturb only ALREADY-NONZERO free modes: waking a zero one changes
        # mpol/ntor (the preset follows the boundary) and we would be comparing
        # two different solver grids instead of two solver versions
        d_rc = rng.normal(size=rc.shape) * mr * (rc != 0)
        d_zs = rng.normal(size=zs.shape) * mz * (zs != 0)
        scale = max(np.abs(d_rc).max(), np.abs(d_zs).max())
        d_rc, d_zs = d_rc / scale, d_zs / scale      # max-coefficient normalized
        np.savez(DIRS / f"{r['key']}.dir.npz", d_rc=d_rc, d_zs=d_zs, h=h)

        for tag, s in (("base", 0.0), ("plus", +h), ("minus", -h)):
            bb = json.loads(json.dumps(b))
            bb["r_cos"] = (rc + s * d_rc).tolist()
            bb["z_sin"] = (zs + s * d_zs).tolist()
            surf = srf.SurfaceRZFourier.model_validate(bb)
            settings = vs.create_vmec_settings_from_preset(
                boundary=surf, settings=vs.VmecPresetSettings(fidelity=fidelity))
            indata = vu.build_vmecpp_indata(
                mhd_parameters=imp.boundary_to_ideal_mhd_parameters(surf),
                boundary=surf, vmec_settings=settings)
            (DIRS / f"{r['key']}.{tag}.indata.json").write_text(
                indata.model_dump_json())
        index.append({"key": r["key"], "stratum": r.get("stratum"), "h": h})
        print(f"dirs-dumped {r['key']} ({r.get('stratum')})", flush=True)
    (DIRS / "index.json").write_text(json.dumps(index, indent=2))
    print(f"STAGE_DIRS_DUMP_OK {len(index)}", flush=True)


def stage_dirs_solve(side: str) -> None:
    """side='pinned' (0.4.11, in-image) or 'sidecar' (0.7.1, sidecar venv)."""
    import vmecpp

    index = json.loads((DIRS / "index.json").read_text())
    for row in index:
        for tag in ("base", "plus", "minus"):
            dst = DIRS / f"{row['key']}.{tag}.{side}.npz"
            if dst.exists():
                continue
            raw = json.loads((DIRS / f"{row['key']}.{tag}.indata.json").read_text())
            t0 = time.time()
            try:
                out = vmecpp.run(vmecpp.VmecInput.model_validate(raw),
                                 max_threads=1, verbose=False)
            except Exception as e:
                print(f"FAIL {row['key']}.{tag}.{side}: "
                      f"{type(e).__name__}: {e}", flush=True)
                continue
            w = out.wout
            arrays = {f: np.asarray(getattr(w, f)) for f in FIELDS
                      if hasattr(w, f)}
            for extra in ("ns", "mpol", "ntor", "nfp", "lasym"):
                if hasattr(w, extra):
                    arrays[extra] = np.asarray(getattr(w, extra))
            np.savez(dst, **arrays)
            print(f"solved {row['key']}.{tag}.{side} in "
                  f"{time.time() - t0:.1f}s", flush=True)
    print(f"STAGE_DIRS_SOLVE_OK {side}", flush=True)


def stage_dirs_compare() -> None:
    index = json.loads((DIRS / "index.json").read_text())
    table = []
    for row in index:
        key, h = row["key"], row["h"]
        rec: dict = {"key": key, "stratum": row.get("stratum"), "h": h}
        for side in ("pinned", "sidecar"):
            try:
                z = {t: dict(np.load(DIRS / f"{key}.{t}.{side}.npz",
                                     allow_pickle=True))
                     for t in ("base", "plus", "minus")}
                m = {t: _metrics(z[t]) for t in z}
            except Exception as e:
                rec[side] = {"error": f"{type(e).__name__}: {e}"}
                continue
            rec[side] = {
                "L_base": m["base"]["min_L_gradB"],
                "dL": (m["plus"]["min_L_gradB"] - m["minus"]["min_L_gradB"]) / (2 * h),
                "diota": (m["plus"]["edge_iota_over_nfp"]
                          - m["minus"]["edge_iota_over_nfp"]) / (2 * h),
            }
        p, s = rec.get("pinned", {}), rec.get("sidecar", {})
        if "dL" in p and "dL" in s:
            rec["dL_rel_disagreement"] = abs(s["dL"] - p["dL"]) / max(abs(p["dL"]), 1e-30)
            rec["cos_sign_agrees"] = (np.sign(p["dL"]) == np.sign(s["dL"])).item()
            print(f"{key} ({rec['stratum']}): dL pinned={p['dL']:+.5g} "
                  f"sidecar={s['dL']:+.5g} rel={rec['dL_rel_disagreement']:.3g} "
                  f"sign_agrees={rec['cos_sign_agrees']}", flush=True)
        table.append(rec)
    (DIRS / "grad_agreement.json").write_text(json.dumps(table, indent=2))
    ok = [t for t in table if "dL_rel_disagreement" in t]
    print("GRAD_AGREEMENT " + json.dumps({
        "n": len(ok),
        "rel_median": float(np.median([t["dL_rel_disagreement"] for t in ok]))
        if ok else None,
        "rel_max": float(np.max([t["dL_rel_disagreement"] for t in ok]))
        if ok else None,
        "sign_agreement": sum(t["cos_sign_agrees"] for t in ok)}), flush=True)



def stage_dirs_compare2() -> None:
    """Same cross-solver comparison, but with the min-kink taken OUT of the
    stencil.

    dirs-compare finite-differences min L-gradB itself, and that metric is a MIN
    over a theta-phi grid: it is only piecewise smooth, and the argmin switches
    under a finite step. Measured consequence: the PINNED dL changes by 1.5-5x
    between h=1e-4 and h=3e-4, i.e. the stencil is not step-size converged, so a
    cross-solver difference cannot be attributed to the solver.

    Here the analytic d(L)/d(wout arrays) is evaluated ONCE at each side's base
    point (Plan-2's differentiable lgradb_jax) and contracted against finite
    differences of the WOUT ARRAYS, which are smooth in the boundary. Same trick
    as the FD qi probe. Uses the stencil solves already on disk — no new solves.
    """
    import jax
    import jax.numpy as jnp
    from experiments.diffscore import lgradb_jax as lg

    ARR = ["rmnc", "zmns", "gmnc", "bmnc", "bsupumnc", "bsupvmnc"]
    index = json.loads((DIRS / "index.json").read_text())
    table = []
    for row in index:
        key, h = row["key"], row["h"]
        rec: dict = {"key": key, "stratum": row.get("stratum"), "h": h}
        for side in ("pinned", "sidecar"):
            try:
                z = {t: dict(np.load(DIRS / f"{key}.{t}.{side}.npz",
                                     allow_pickle=True))
                     for t in ("base", "plus", "minus")}
            except Exception as e:
                rec[side] = {"error": f"{type(e).__name__}: {e}"}
                continue
            b = z["base"]
            ns = int(b["ns"]); mpol, ntor = int(b["mpol"]), int(b["ntor"])
            nfp = int(b["nfp"]); lasym = bool(b["lasym"])

            def f(*arrs):
                d = dict(zip(ARR, arrs))
                return lg.min_normalized_l_grad_b(
                    d["rmnc"], d["zmns"], d["gmnc"], d["bmnc"], d["bsupumnc"],
                    d["bsupvmnc"], b["xm"], b["xn"], b["xm_nyq"], b["xn_nyq"],
                    ns, float(b["Aminor_p"]), mpol, ntor, nfp, lasym)

            base_args = [jnp.asarray(b[k]) for k in ARR]
            val, grads = jax.value_and_grad(f, argnums=tuple(range(len(ARR))))(
                *base_args)
            dL = 0.0
            for k, g in zip(ARR, grads):
                if z["plus"][k].shape != z["minus"][k].shape:
                    dL = None
                    break
                dL += float(np.sum(np.asarray(g)
                                   * (z["plus"][k] - z["minus"][k]) / (2 * h)))
            rec[side] = {"L_base": float(val), "dL_contracted": dL}
        p, s = rec.get("pinned", {}), rec.get("sidecar", {})
        if p.get("dL_contracted") is not None and s.get("dL_contracted") is not None:
            rec["rel"] = abs(s["dL_contracted"] - p["dL_contracted"]) / max(
                abs(p["dL_contracted"]), 1e-30)
            rec["sign_agrees"] = bool(
                np.sign(p["dL_contracted"]) == np.sign(s["dL_contracted"]))
            print(f"{key} ({rec['stratum']}) h={h:g}: dL pinned="
                  f"{p['dL_contracted']:+.6g} sidecar={s['dL_contracted']:+.6g} "
                  f"rel={rec['rel']:.3g} sign_agrees={rec['sign_agrees']}",
                  flush=True)
        table.append(rec)
    (DIRS / "grad_agreement_contracted.json").write_text(json.dumps(table, indent=2))
    ok = [t for t in table if "rel" in t]
    print("GRAD_AGREEMENT_CONTRACTED " + json.dumps({
        "n": len(ok),
        "rel_median": float(np.median([t["rel"] for t in ok])) if ok else None,
        "rel_max": float(np.max([t["rel"] for t in ok])) if ok else None,
        "sign_agreement": sum(t["sign_agrees"] for t in ok)}), flush=True)

if __name__ == "__main__":
    sys.exit(main())
