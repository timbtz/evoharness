"""Second-order NAE (nae2) boundary generator — wiki/03-leads.md L5 + L7.

Runs INSIDE the evoharness-stellar-eval image (needs constellaration + pyQIC;
the host venv has neither):

  docker run --rm --cpus 1 --memory 3g --user $(id -u):$(id -g) -e HOME=/tmp \
      -v "$PWD:/work" -w /work evoharness-stellar-eval \
      python -m evoharness.tasks.stellar_p2.proposals.near_axis

Pipeline per (nfp, torsion) design point:
  1. constellaration's first-order QI constructor (near_axis_configuration.
     generate) with the iota barrier set to the P2 requirement
     edge_iota/nfp >= 0.25 (axis iota ~ edge iota at these betas).
     Torsion is pre-scanned per nfp because the barrier is soft (weight 100)
     and undershoots at low torsion (measured: torsion 0.3 @ nfp3 ->
     iota 0.435 vs target 0.79).
  2. pyQIC's construct_qi_r2 (Rodriguez-Plunk second-order QI conditions):
     optimizes d-bar Fourier content for 2nd-order consistency, derives the
     X2c/X2s cross-section terms, re-inits the config at order='r2'. The
     boundary map (Frenet_to_cylindrical) then carries the r^2 displacement
     — the "second-order axis content" lever the first-order arm cannot reach.
     UPSTREAM BUG avoided: construct_qi_r2's d_over_curvature_cvals seeding
     uses `==` where `=` was meant (qic 0.3.4 calculate_r2.py:459/467), so the
     Qic object here is always built with an explicit length-4
     d_over_curvature_cvals, which skips the broken branch.
  3. X2s_in sweep {0, +X, -X}: the only exposed free second-order DOF; it
     shifts where the r^2 shaping concentrates (bean/inboard region target).
  4. Fourier fit at moderate mode content (mpol 8 / ntor 10) plus one
     deliberately truncated variant (4/4) to measure the truncation cost the
     constructor's smoothing usually imposes.

Outputs: <out>/candidates/<name>.json  ({"boundary": ..., "meta": ...})
         <out>/index.json              (scan table + per-candidate meta)
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from evoharness.paths import RUNS_DIR

warnings.filterwarnings("ignore")

IOTA_MARGIN = 1.06          # aim 6% above the P2 bound edge_iota/nfp >= 0.25
MIRROR_RATIO = 0.19         # P2 bound is 0.20 at the edge; on-axis well depth
MAX_ELONGATION = 4.8        # P2 bound 5.0
MINOR_RADIUS = 0.1          # A = R0/r = 10.0 = the P2 bound (R0 = 1)


def _patch_qic() -> None:
    """qic 0.3.4 breaks when d_over_curvature_cvals arrives as an ndarray
    (init_axis: `ndarray == []` broadcast ValueError) — construct_qi_r2's own
    final re-init does exactly that. Coerce to a plain list at the boundary.
    Process-local; the image on disk is never modified."""
    import qic
    if getattr(qic.Qic, "_evoh_patched", False):
        return
    orig = qic.Qic.__init__

    def patched(self, *args, **kwargs):
        # qic compares many list-kwargs with `== []`, which raises on
        # ndarrays (B0_vals, d_*vals, B2*_vals, ...). construct_qi_r2's final
        # re-init passes its own ndarray state back in — coerce all of them.
        for k, v in list(kwargs.items()):
            if isinstance(v, np.ndarray):
                kwargs[k] = v.ravel().tolist()
        return orig(self, *args, **kwargs)

    qic.Qic.__init__ = patched

    # set_dofs() assigns d_over_curvature_cvals as an ndarray directly (no
    # __init__), so init_axis must also be guarded.
    orig_axis = qic.Qic.init_axis

    def patched_axis(self, *args, **kwargs):
        v = getattr(self, "d_over_curvature_cvals", [])
        if not isinstance(v, list):
            self.d_over_curvature_cvals = np.asarray(v).ravel().tolist()
        return orig_axis(self, *args, **kwargs)

    qic.Qic.init_axis = patched_axis
    qic.Qic._evoh_patched = True


def _boundary_dict(stel, nfp: int, minor_radius: float,
                   mpol: int, ntor: int, ntheta: int = 24) -> dict:
    """Fourier boundary at the largest workable minor radius <= requested,
    SPECTRALLY CONDENSED via constellaration's smooth_and_set_max_mode_numbers
    (simsopt least-squares refit). A raw to_Fourier boundary is VMEC-fatal
    ("solver failed during the first iterations... not spectrally condensed",
    measured on all three r1 baselines 2026-08-15/16).
    High-torsion axes also fold the Frenet frame at r=0.1 (brentq sign error);
    step down and record the radius actually used (aspect = 1/r at R0=1)."""
    import qic.util
    from constellaration.geometry import surface_rz_fourier as srf
    from constellaration.mhd import near_axis_configuration as nac
    last: Exception | None = None
    for r in (minor_radius, 0.085, 0.07, 0.055):
        try:
            R2D, Z2D, _ = stel.Frenet_to_cylindrical(r=r, ntheta=ntheta)
            r_cos, _, _, z_sin = qic.util.to_Fourier(
                R_2D=R2D, Z_2D=Z2D, nfp=nfp,
                mpol=ntheta // 2, ntor=15, lasym=False)
            raw = srf.SurfaceRZFourier(
                r_cos=r_cos.T, z_sin=z_sin.T, n_field_periods=nfp,
                is_stellarator_symmetric=True)
            smooth = nac.smooth_and_set_max_mode_numbers(
                raw, max_poloidal_mode=mpol, max_toroidal_mode=ntor)
            bd = {"r_cos": np.asarray(smooth.r_cos).tolist(),
                  "z_sin": np.asarray(smooth.z_sin).tolist(),
                  "r_sin": None, "z_cos": None, "n_field_periods": nfp,
                  "is_stellarator_symmetric": True}
            return bd, r
        except Exception as e:          # fold/brentq — try a thinner surface
            last = e
    raise RuntimeError(f"no workable minor radius: {last!r}")


def _r1_diag(stel) -> dict:
    return {"iota": float(stel.iota),
            "max_elongation": float(np.max(stel.elongation)),
            "qi_residual_r1": float(stel.min_geo_qi_consistency(order=1))}


def _make_r1(nfp: int, torsion: float, min_iota: float):
    from constellaration.mhd import near_axis_configuration as nac
    cfg = nac.generate(mirror_ratio=MIRROR_RATIO, min_iota=min_iota,
                       max_elongation=MAX_ELONGATION, torsion=torsion,
                       n_field_periods=nfp, max_toroidal_mode=3)
    stel = nac.near_axis_configuration_to_pyqic(cfg)
    return cfg, stel


def _make_r2(cfg, b2c: float, b2s: float):
    """MINIMAL-SHAPING second order (Rodriguez & Plunk arXiv:2409.20328):
    plain order='r2' with scalar B2c/B2s inputs. construct_qi_r2 was measured
    degenerate on these configs (X20/X2c ~ 2000 vs X1c ~ 0.3, B2cQI dev 290,
    folds at every radius); minimal shaping gives X20 ~ 1.7 and a workable
    surface at r=0.07. B2c/B2s are the controlled second-order shaping dial
    (X2c/X2s scale with them); (0,0) is the pure minimal-shaping X20-only
    (Shafranov-type) construction."""
    import qic
    _patch_qic()
    d0 = float(cfg.d_over_curvature) or 0.5
    stel = qic.Qic(omn_method=cfg.omnigeneity_method,
                   p_buffer=cfg.p_buffer, k_buffer=cfg.k_buffer,
                   rc=cfg.r_cos.tolist(), zs=cfg.z_sin.tolist(),
                   nfp=cfg.n_field_periods, B0_vals=cfg.B0_cos.tolist(),
                   nphi=31, omn=True, order="r2", B2c=b2c, B2s=b2s,
                   d_over_curvature_cvals=[d0, 0.0, 0.0, 0.0],
                   d_svals=cfg.d_sin.tolist())
    dev = float(np.max(np.abs(getattr(stel, "B20_variation", np.nan))))
    return stel, dev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS_DIR / "nae2-batch1"))
    ap.add_argument("--nfp", default="3,4,5")
    ap.add_argument("--torsions", default="0.4,0.7,1.0,1.4")
    ap.add_argument("--b2", default="0:0,0.2:0,-0.2:0,0:0.2",
                    help="comma list of B2c:B2s scalar pairs for the r2 arm")
    ap.add_argument("--mpol", type=int, default=5)
    ap.add_argument("--ntor", type=int, default=5)
    ap.add_argument("--pick", default="",
                    help="override torsion per nfp, e.g. 3=0.7,4=0.7,5=1.0")
    ap.add_argument("--all-scanned", action="store_true",
                    help="emit candidates for every scanned (nfp,torsion)")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    nfps = [int(x) for x in a.nfp.split(",")]
    torsions = [float(x) for x in a.torsions.split(",")]
    b2_list = [tuple(float(y) for y in x.split(":")) for x in a.b2.split(",")]
    index: dict = {"scan": [], "candidates": [], "errors": []}

    def flush():
        (out / "index.json").write_text(json.dumps(index, indent=2))

    picks = {int(k): float(v) for k, v in
             (kv.split("=") for kv in a.pick.split(",") if kv)}

    # ---- phase 1: torsion scan per nfp (first order only, cheap) ----------
    chosen: dict[int, tuple[float, object, object]] = {}
    cache: dict[tuple[int, float], tuple[object, object]] = {}
    for nfp in nfps:
        min_iota = 0.25 * nfp * IOTA_MARGIN
        best = None
        scan_torsions = [picks[nfp]] if nfp in picks else torsions
        for tor in scan_torsions:
            t0 = time.time()
            try:
                cfg, stel = _make_r1(nfp, tor, min_iota)
                cache[(nfp, tor)] = (cfg, stel)
                d = _r1_diag(stel)
            except Exception as e:
                index["errors"].append({"stage": "scan", "nfp": nfp,
                                        "torsion": tor, "error": repr(e)[:300]})
                flush()
                continue
            row = {"nfp": nfp, "torsion": tor, "min_iota_target": min_iota,
                   "seconds": round(time.time() - t0, 1), **d}
            index["scan"].append(row)
            flush()
            print(f"scan nfp={nfp} tor={tor}: iota={d['iota']:.3f} "
                  f"elong={d['max_elongation']:.2f} "
                  f"qi={d['qi_residual_r1']:.2e}", flush=True)
            # prefer iota target met AND sane qi, then lowest qi residual
            key = (d["iota"] >= min_iota and d["qi_residual_r1"] < 1.0,
                   -d["qi_residual_r1"])
            if best is None or key > best[0]:
                best = (key, tor, cfg, stel)
        if nfp in picks and (nfp, picks[nfp]) in cache:
            cfg, stel = cache[(nfp, picks[nfp])]
            chosen[nfp] = (picks[nfp], cfg, stel)
        elif best is not None:
            chosen[nfp] = (best[1], best[2], best[3])

    # ---- phase 2: candidate batch -----------------------------------------
    # every scanned (nfp, torsion) point gets the r1/r2 pair — which r1s
    # VMEC accepts is itself a measurement (high-torsion surfaces fail).
    batch_points = (cache.items() if a.all_scanned
                    else (((nfp, tor), (cfg, stel))
                          for nfp, (tor, cfg, stel) in chosen.items()))
    for (nfp, tor), (cfg, stel1) in batch_points:
        tag = f"nfp{nfp}-t{tor:g}"
        # (a) first-order baseline — the control the r2 arms are paired with
        try:
            bd, r_used = _boundary_dict(stel1, nfp, MINOR_RADIUS,
                                        a.mpol, a.ntor)
            meta = {"variant": "r1-base", "nfp": nfp, "torsion": tor,
                    "minor_radius": r_used,
                    "mpol": a.mpol, "ntor": a.ntor, **_r1_diag(stel1)}
            name = f"{tag}-r1"
            (out / "candidates" / f"{name}.json").write_text(
                json.dumps({"boundary": bd, "meta": meta}))
            index["candidates"].append({"name": name, **meta})
            flush()
            print(f"wrote {name}", flush=True)
        except Exception:
            index["errors"].append({"stage": "r1-base", "nfp": nfp,
                                    "error": traceback.format_exc()[-400:]})
            flush()

        # (b) second-order minimal-shaping variants (B2c/B2s dial)
        for b2c, b2s in b2_list:
            t0 = time.time()
            name = f"{tag}-r2-c{b2c:g}s{b2s:g}"
            try:
                stel2, dev = _make_r2(cfg, b2c, b2s)
                bd, r_used = _boundary_dict(stel2, nfp, MINOR_RADIUS,
                                            a.mpol, a.ntor)
                meta = {"variant": "r2-min", "nfp": nfp, "torsion": tor,
                        "B2c": b2c, "B2s": b2s, "B20_variation_max": dev,
                        "iota": float(stel2.iota),
                        "X20_max": float(np.max(np.abs(stel2.X20))),
                        "minor_radius": r_used,
                        "mpol": a.mpol, "ntor": a.ntor,
                        "seconds": round(time.time() - t0, 1)}
                (out / "candidates" / f"{name}.json").write_text(
                    json.dumps({"boundary": bd, "meta": meta}))
                index["candidates"].append({"name": name, **meta})
                flush()
                print(f"wrote {name} (B20var={dev:.1f}, r={r_used}, "
                      f"{meta['seconds']}s)", flush=True)
                # (c) mode-content twins of the plain minimal-shaping arm
                if b2c == 0.0 and b2s == 0.0:
                    for mp, nt, suffix in ((3, 3, "trunc"), (8, 10, "hi")):
                        bdx, rx = _boundary_dict(stel2, nfp, MINOR_RADIUS,
                                                 mp, nt)
                        metax = {**meta, "variant": f"r2-{suffix}",
                                 "mpol": mp, "ntor": nt, "minor_radius": rx}
                        namex = f"{tag}-r2-{suffix}"
                        (out / "candidates" / f"{namex}.json").write_text(
                            json.dumps({"boundary": bdx, "meta": metax}))
                        index["candidates"].append({"name": namex, **metax})
                        flush()
                        print(f"wrote {namex}", flush=True)
            except Exception:
                index["errors"].append({"stage": "r2", "nfp": nfp,
                                        "b2": [b2c, b2s],
                                        "error": traceback.format_exc()[-400:]})
                flush()
                print(f"FAILED {name}", flush=True)

    print(f"done: {len(index['candidates'])} candidates, "
          f"{len(index['errors'])} errors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
