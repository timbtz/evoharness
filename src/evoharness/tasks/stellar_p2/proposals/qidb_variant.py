"""Route (c): build P2 boundaries from the near-axis QI database's OWN
first-order solution, with no pyQIC object at all.

Route (b) (experiments/qidb_boundary.py) transfers only the design
(axis + B0 + elongation) and lets pinned qic 0.3.4 re-solve sigma.  That was
measured to fail structurally on 2026-08-16: the database is a HALF-HELICITY
(2,3)-flattening-class family built in the SIGNED Frenet frame, while qic 0.3.4
implements the helicity-0 Plunk-Landreman-Helander class -- the re-solve landed
on a different branch (elongation 9.5-25 vs the design's 4.3, first-order QI
consistency 1e7-1e12 vs a healthy 3e-2, and violently nphi-dependent), and both
boundaries were VMEC-fatal in 3 s.

Route (c) bypasses the solve.  The per-sweep database JSON stores the complete
first-order solution on a 201-point Boozer grid: sigma, d, dbar, alpha, kappa,
tau, B0, varphi, d_l_d_varphi.  With Bbar = 1, sG = 1 the standard first-order
relations (qic/calculate_r1.py:254-256, with B1c = B0 d cos(alpha),
B1s = B0 d sin(alpha) for a QI field) collapse to

    X1(theta, phi) = dbar * cos(theta - alpha)
    Y1(theta, phi) = [ sin(theta - alpha) + sigma * cos(theta - alpha) ] / (B0*dbar)

and the surface is  x = x0(l) + r X1 n_hat(l) + r Y1 b_hat(l)  in the signed
Frenet frame, which we obtain by integrating the same kappa(l), tau(l) the
database used.  d and dbar are divided by sqrt(2) (the database JSON uses
Plunk's normalisation, pyQIC/Landreman's is sqrt(2) smaller -- see
Paper_scripts/Scripts/utils.py::make_stel_object).

Runs INSIDE the eval image (needs qic.util.to_Fourier + constellaration):

  docker run --rm --cpus 1 --memory 3g --user $(id -u):$(id -g) -e HOME=/tmp \
      -v "$PWD:/work" -v <scratch>:/sp -w /work evoharness-stellar-eval \
      python -m evoharness.tasks.stellar_p2.proposals.qidb_variant \
      --db-dir /sp/dbjson
"""
from __future__ import annotations

import argparse
import json
import traceback
import warnings
from pathlib import Path

import numpy as np
from evoharness.paths import RUNS_DIR

warnings.filterwarnings("ignore")


def load_db_entry(path, loader_id):
    raw = json.load(open(path))
    item = raw[loader_id]
    return {k: (np.array(v, dtype=float) if isinstance(v, list) else v)
            for k, v in item}


def frenet_frame(N, k1, k2, t0, t1, t2, n_int=8192):
    """Signed Frenet-Serret frame of the database's axis, on l in [0, 2pi)."""
    from scipy.integrate import solve_ivp

    def rhs(l, y):
        t, kv, bv = y[3:6], y[6:9], y[9:12]
        h = N * l / 2.0
        k = 2.0 * k1 * np.sin(h) ** 2 * np.cos(h) ** 3 * (1.0 + 2.0 * k2 * np.cos(N * l))
        tt = t0 + t1 * np.cos(N * l) + t2 * np.cos(2.0 * N * l)
        return np.concatenate([t, k * kv, -k * t + tt * bv, -tt * kv])

    y0 = np.zeros(12)
    y0[3] = y0[7] = y0[11] = 1.0
    ls = np.linspace(0.0, 2.0 * np.pi, n_int, endpoint=False)
    sol = solve_ivp(rhs, (0.0, 2.0 * np.pi), y0, t_eval=ls,
                    rtol=1e-12, atol=1e-14, method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    x = sol.y[0:3].T
    closure = float(np.linalg.norm(x[-1] + (x[-1] - x[-2]) - x[0]))
    n = sol.y[6:9].T
    b = sol.y[9:12].T
    return ls, x, n, b, closure


def principal_frame(x):
    """Centroid + principal axes; returns (centre, e1, e2, zax)."""
    c = x.mean(axis=0)
    y = x - c
    _, v = np.linalg.eigh(y.T @ y)
    zax, e1 = v[:, 0], v[:, 2]
    e2 = np.cross(zax, e1)
    # orient so that the toroidal angle increases with l
    px, py = y @ e1, y @ e2
    ph = np.unwrap(np.arctan2(py, px))
    if ph[-1] < ph[0]:
        e2 = -e2
        zax = -zax
    return c, e1, e2, zax


def to_cyl(pts, c, e1, e2, zax):
    y = pts - c
    px, py, pz = y @ e1, y @ e2, y @ zax
    return np.hypot(px, py), np.unwrap(np.arctan2(py, px)), pz


def build_surface(cand, db_entry, r, ntheta=24, nphi_out=61, dbar_floor=1e-3):
    inp = cand["input_features"]
    N = int(cand["db_ref"]["N_field_periods"])
    ls, x0, nh, bh, closure = frenet_frame(
        N, inp["kappa_s1"], inp["kappa_s2"],
        inp["tau_c0"], inp["tau_c1"], inp["tau_c2"])
    c, e1, e2, zax = principal_frame(x0)

    # database first-order solution on ONE field period (drop the repeated end)
    varphi = db_entry["varphi"]
    dl_dvp = db_entry["d_l_d_varphi"]
    sigma = db_entry["sigma"]
    alpha = db_entry["alpha"]
    dbar = db_entry["dbar"] / np.sqrt(2.0)
    B0 = db_entry["B0"]
    # arc length of each grid point (upstream utils.py does the same cumtrapz)
    ell = np.concatenate([[0.0], np.cumsum(
        0.5 * (dl_dvp[1:] + dl_dvp[:-1]) * np.diff(varphi))])
    m = len(varphi) - 1                       # drop the duplicated endpoint
    ell, sigma, alpha, dbar, B0 = ell[:m], sigma[:m], alpha[:m], dbar[:m], B0[:m]

    # frame interpolated onto the database's arc-length stations
    def interp_vec(a):
        return np.stack([np.interp(ell, ls, a[:, k], period=2 * np.pi)
                         for k in range(3)], axis=1)

    X0 = interp_vec(x0)
    NH = interp_vec(nh)
    BH = interp_vec(bh)

    # first-order displacement
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    ch = np.cos(theta[:, None] - alpha[None, :])
    sh = np.sin(theta[:, None] - alpha[None, :])
    db_safe = np.where(np.abs(dbar) < dbar_floor,
                       np.sign(dbar + 1e-30) * dbar_floor, dbar)
    X1 = dbar[None, :] * ch
    Y1 = (sh + sigma[None, :] * ch) / (B0[None, :] * db_safe[None, :])
    pts = (X0[None, :, :]
           + r * X1[:, :, None] * NH[None, :, :]
           + r * Y1[:, :, None] * BH[None, :, :])

    # cylindrical, then resample onto a uniform toroidal grid over one period
    R0c, phi0c, Z0c = to_cyl(X0, c, e1, e2, zax)
    Rmaj = float(np.mean(np.hypot((x0 - c) @ e1, (x0 - c) @ e2)))
    phi_start = float(phi0c[0])
    phig = np.linspace(phi_start, phi_start + 2.0 * np.pi / N, nphi_out,
                       endpoint=False)
    R_2D = np.zeros((ntheta, nphi_out))
    Z_2D = np.zeros((ntheta, nphi_out))
    for j in range(ntheta):
        Rj, phij, Zj = to_cyl(pts[j], c, e1, e2, zax)
        o = np.argsort(phij)
        R_2D[j] = np.interp(phig, phij[o], Rj[o], period=2 * np.pi)
        Z_2D[j] = np.interp(phig, phij[o], Zj[o], period=2 * np.pi)
    diag = {
        "closure_err": closure,
        "R_major_axis": Rmaj,
        "axis_R_min_over_R0": float(R0c.min() / Rmaj),
        "surface_R_min": float(R_2D.min()),
        "surface_R_max": float(R_2D.max()),
        "surface_Z_absmax": float(np.abs(Z_2D).max()),
        "dbar_min_abs": float(np.abs(dbar).min()),
        "sigma_absmax": float(np.abs(sigma).max()),
        "B0_max": float(B0.max()), "B0_min": float(B0.min()),
        "mirror_on_axis": float((B0.max() - B0.min()) / (B0.max() + B0.min())),
        "iota_db": float(db_entry["iota"]),
        "aspect_estimate": float(Rmaj / r),
        # first-order edge mirror straight from the stored d
        "edge_mirror_firstorder": float(
            ((B0 * (1 + r * np.abs(db_entry["d"][:m] / np.sqrt(2)))).max()
             - (B0 * (1 - r * np.abs(db_entry["d"][:m] / np.sqrt(2)))).min())
            / ((B0 * (1 + r * np.abs(db_entry["d"][:m] / np.sqrt(2)))).max()
               + (B0 * (1 - r * np.abs(db_entry["d"][:m] / np.sqrt(2)))).min())),
    }
    # elongation of the r-surface cross-sections (max over the period)
    el = []
    for k in range(nphi_out):
        rr = R_2D[:, k] - R_2D[:, k].mean()
        zz = Z_2D[:, k] - Z_2D[:, k].mean()
        cov = np.cov(np.stack([rr, zz]))
        w = np.linalg.eigvalsh(cov)
        el.append(np.sqrt(max(w) / max(min(w), 1e-30)))
    diag["elongation_max_estimate"] = float(np.max(el))
    return R_2D, Z_2D, diag, N


def fourier_boundary(R_2D, Z_2D, nfp, mpol, ntor, ntor_fit=15):
    import qic.util
    from constellaration.geometry import surface_rz_fourier as srf
    from constellaration.mhd import near_axis_configuration as nac
    ntheta = R_2D.shape[0]
    r_cos, _, _, z_sin = qic.util.to_Fourier(
        R_2D=R_2D, Z_2D=Z_2D, nfp=nfp, mpol=ntheta // 2, ntor=ntor_fit,
        lasym=False)
    raw = srf.SurfaceRZFourier(r_cos=r_cos.T, z_sin=z_sin.T,
                               n_field_periods=nfp, is_stellarator_symmetric=True)
    sm = nac.smooth_and_set_max_mode_numbers(
        raw, max_poloidal_mode=mpol, max_toroidal_mode=ntor)
    return {"r_cos": np.asarray(sm.r_cos).tolist(),
            "z_sin": np.asarray(sm.z_sin).tolist(),
            "r_sin": None, "z_cos": None,
            "n_field_periods": int(nfp), "is_stellarator_symmetric": True}


def rescale_to_major_radius(bd, target=1.0):
    a = np.array(bd["r_cos"], dtype=float)
    R00 = a[0, a.shape[1] // 2]
    s = target / R00
    bd["r_cos"] = (a * s).tolist()
    bd["z_sin"] = (np.array(bd["z_sin"], dtype=float) * s).tolist()
    return bd, float(s)


def blend_to_ellipse(bd, lam):
    out = json.loads(json.dumps(bd))
    for key in ("r_cos", "z_sin"):
        a = np.array(out[key], dtype=float)
        if a.shape[0] > 2:
            a[2:, :] *= lam
        out[key] = a.tolist()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS_DIR / "qidb-route-c"))
    ap.add_argument("--db-dir", default="/sp/dbjson")
    ap.add_argument("--candidates", default=str(RUNS_DIR / "qi-database-mine/candidates.json"))
    ap.add_argument("--buckets", default="B_geo_and_epseff2pct,B2_geo_and_epseff1pct")
    ap.add_argument("--per-bucket", type=int, default=1)
    ap.add_argument("--radii", default="0.1")
    ap.add_argument("--mpol-ntor", default="3:3,5:5")
    ap.add_argument("--ntheta", type=int, default=24)
    ap.add_argument("--nphi-out", type=int, default=61)
    ap.add_argument("--blends", default="")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = json.load(open(a.candidates))["candidates"]
    picked = []
    for b in [x.strip() for x in a.buckets.split(",") if x.strip()]:
        picked += [c for c in cands if c["bucket"] == b][: a.per_bucket]

    radii = [float(x) for x in a.radii.split(",")]
    mns = [tuple(int(v) for v in p.split(":")) for p in a.mpol_ntor.split(",")]
    blends = [None] + [float(x) for x in a.blends.split(",") if x.strip()]

    index = []
    for c in picked:
        src = Path(a.db_dir) / Path(c["db_ref"]["source_json"]).name
        entry = load_db_entry(src, c["db_ref"]["loader_config_id"])
        tag = (f"{c['bucket']}-N{c['db_ref']['N_field_periods']}"
               f"-cfg{c['db_ref']['configurationIndex']}")
        # sanity: the JSON entry must be the configuration we think it is
        chk = {"nfp": int(entry["nfp"]),
               "id_within": int(entry["configurationIndex"]),
               "iota_json": float(entry["iota"]),
               "iota_db": c["p2_proxies"]["iota_near_axis"]}
        print(json.dumps({"check": tag, **chk}), flush=True)
        for r in radii:
            try:
                R2, Z2, diag, N = build_surface(c, entry, r, a.ntheta, a.nphi_out)
            except Exception as e:
                index.append({"name": f"{tag}-r{r:g}", "ok": False,
                              "error": f"{e!r}"[:300],
                              "trace": traceback.format_exc()[-400:]})
                print(json.dumps(index[-1] | {"trace": ""}), flush=True)
                continue
            for mpol, ntor in mns:
                for bl in blends:
                    name = (f"{tag}-r{r:g}-m{mpol}n{ntor}"
                            + (f"-bl{bl:g}" if bl is not None else ""))
                    rec = {"name": name, "r": r, "mpol": mpol, "ntor": ntor,
                           "blend": bl}
                    try:
                        bd = fourier_boundary(R2, Z2, N, mpol, ntor)
                        if bl is not None:
                            bd = blend_to_ellipse(bd, bl)
                        bd, s = rescale_to_major_radius(bd)
                        meta = {"candidate": c["db_ref"], "bucket": c["bucket"],
                                "input_features": c["input_features"],
                                "near_axis_prediction": c["p2_proxies"],
                                "db_metrics": c["db_metrics"],
                                "route": "c (database's own first-order solution, "
                                         "signed Frenet frame; no pyQIC solve)",
                                "r": r, "mpol": mpol, "ntor": ntor, "blend": bl,
                                "rescale_to_R0_1": s, "surface_diag": diag,
                                "json_check": chk}
                        (out / f"{name}.json").write_text(
                            json.dumps({"boundary": bd, "meta": meta}, indent=1))
                        rec["ok"] = True
                        rec["diag"] = diag
                    except Exception as e:
                        rec["ok"] = False
                        rec["error"] = f"{e!r}"[:300]
                    index.append(rec)
                    print(json.dumps(rec), flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"BUILT {sum(1 for r in index if r.get('ok'))}/{len(index)} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
