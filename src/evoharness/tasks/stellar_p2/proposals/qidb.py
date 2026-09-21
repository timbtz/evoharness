"""Route (b): build P2 boundaries from near-axis QI database candidates using
the PINNED stack only (qic 0.3.4 + constellaration 0.2.6) -- no pyQIC fork.

wiki/07 section 7 route (b): the database stores its configurations as
{kappa(l), tau(l), rho(phi), B0(l)} closed forms plus a Frenet-frame solution
that only the SebereX/pyQIC@change_structure fork can ingest.  Instead of the
fork we transfer the *design*:

  1. integrate the signed Frenet-Serret system with the stored kappa/tau
     Fourier inputs -> the magnetic axis in R^3,
  2. project it into cylindrical coordinates and fit R0(phi), Z0(phi) in the
     qic convention  R0 = sum_i rc[i] cos(i*nfp*phi),  Z0 = sum_i zs[i] sin(...),
     normalised to major radius 1,
  3. B0_vals straight from the database's B0 model
     B0 = 1 + D[(1+lB)cos(N l) + (1/4-2 lB)cos(2N l) - lB cos(3N l)],
  4. d_over_curvature_cvals from the database's elongation model:
     qic's own init_axis defines etabar^2/kappa^2 = (B0/Bbar) * dbar^2, and the
     database's ellipticity input is rho = ebar + (1+sigma^2)/ebar, so with
     sigma neglected  dbar = sqrt(ebar/B0),  ebar = (rho - sqrt(rho^2-4))/2.
     NOTE the stored RhoForm carries a pi phase on the first harmonic:
     rho = rc0 - rc1 cos(N phi) + rc2 cos(2N phi).
  5. let the pinned qic re-solve sigma / alpha / iota (order='r1'),
  6. Frenet_to_cylindrical at r -> to_Fourier -> spectral condensation
     (mandatory, wiki/05) -> boundary dict.

Runs INSIDE the eval image:

  docker run --rm --cpus 1 --memory 3g --user $(id -u):$(id -g) -e HOME=/tmp \
      -v "$PWD:/work" -w /work evoharness-stellar-eval \
      python -m evoharness.tasks.stellar_p2.proposals.qidb

Outputs <out>/<name>.json = {"boundary": ..., "meta": ...} plus <out>/index.json.
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

CANDIDATES = str(RUNS_DIR / "qi-database-mine/candidates.json")


# --------------------------------------------------------------------------
# 1-2. axis: Frenet-Serret integration -> cylindrical Fourier (qic convention)
# --------------------------------------------------------------------------
def axis_from_kappa_tau(N, k1, k2, t0, t1, t2, n_int=4096, ntor=6, ng=2048):
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
        raise RuntimeError(f"FS integration failed: {sol.message}")
    x = sol.y[0:3].T
    closure = float(np.linalg.norm(x[-1] + (x[-1] - x[-2]) - x[0]))
    x = x - x.mean(axis=0)
    # principal frame: least-inertia direction is the torus symmetry axis
    _, v = np.linalg.eigh(x.T @ x)
    zax, e1 = v[:, 0], v[:, 2]
    e2 = np.cross(zax, e1)
    px, py, pz = x @ e1, x @ e2, x @ zax
    R = np.hypot(px, py)
    phi = np.unwrap(np.arctan2(py, px))
    if phi[-1] < phi[0]:            # enforce increasing toroidal angle
        phi, pz = -phi, -pz
    # resample on a uniform grid whose ORIGIN is the l=0 point (B0 max, kappa=0,
    # and the stellarator-symmetry point of this construction)
    phig = np.linspace(phi[0], phi[0] + 2.0 * np.pi, ng, endpoint=False)
    o = np.argsort(phi)
    Rg = np.interp(phig, phi[o], R[o], period=2 * np.pi)
    Zg = np.interp(phig, phi[o], pz[o], period=2 * np.pi)
    Rf, Zf = np.fft.rfft(Rg) / ng, np.fft.rfft(Zg) / ng
    R0 = float(Rf[0].real)
    # qic index i <-> absolute toroidal mode i*nfp
    idx = [i * N for i in range(ntor + 1)]
    if max(idx) >= len(Rf):
        raise RuntimeError("grid too coarse for requested ntor")
    rc = np.array([Rf[0].real if i == 0 else 2.0 * Rf[i].real for i in idx]) / R0
    zs = np.array([0.0 if i == 0 else -2.0 * Zf[i].imag for i in idx]) / R0
    # stellarator-symmetry QA: the wrong-parity content must be negligible
    rs = np.array([0.0 if i == 0 else -2.0 * Rf[i].imag for i in idx]) / R0
    zc = np.array([Zf[0].real if i == 0 else 2.0 * Zf[i].real for i in idx]) / R0
    asym = float(max(np.abs(rs).max(), np.abs(zc).max()))
    return {
        "rc": rc.tolist(), "zs": zs.tolist(),
        "R0_raw": R0, "closure_err": closure, "asymmetry": asym,
        "R_min_over_R0": float(Rg.min() / R0),
        "helical_amp": float(max(np.abs(rc[1:]).max(), np.abs(zs[1:]).max())),
        "axis_length_over_R0": float(2.0 * np.pi / R0),
    }


# --------------------------------------------------------------------------
# 3-4. B0 and dbar from the database's closed forms
# --------------------------------------------------------------------------
def b0_vals(Delta, lamB):
    return [1.0, Delta * (1.0 + lamB), Delta * (0.25 - 2.0 * lamB), -Delta * lamB]


def dbar_cvals(N, Delta, lamB, r0, r1, r2, nharm=4, ng=1024, rho_sign=-1.0):
    """Cosine coefficients of dbar(varphi) = sqrt(ebar/B0).

    rho_sign = -1 reproduces the stored RhoForm phase (rho1 enters with a minus).
    """
    x = np.linspace(0.0, 2.0 * np.pi / N, ng, endpoint=False)
    B0 = (1.0 + Delta * ((1.0 + lamB) * np.cos(N * x)
                         + (0.25 - 2.0 * lamB) * np.cos(2 * N * x)
                         - lamB * np.cos(3 * N * x)))
    rho = r0 + rho_sign * r1 * np.cos(N * x) + r2 * np.cos(2 * N * x)
    if np.any(rho < 2.0):
        raise RuntimeError(f"rho < 2 somewhere (min {rho.min():.4f}) -- invalid ellipse")
    ebar = 0.5 * (rho - np.sqrt(rho**2 - 4.0))
    dbar = np.sqrt(ebar / B0)
    c = [float(dbar.mean())]
    for n in range(1, nharm + 1):
        c.append(float(2.0 * np.mean(dbar * np.cos(n * N * x))))
    resid = dbar - sum(c[n] * np.cos(n * N * x) for n in range(len(c)))
    return c, float(np.abs(resid).max() / np.abs(dbar).max())


# --------------------------------------------------------------------------
# 5. pinned qic object
# --------------------------------------------------------------------------
def make_stel(N, rc, zs, B0v, dcv, omn_method="non-zone-smoother",
              k_buffer=2, p_buffer=3, nphi=131, order="r1", **kw):
    import qic
    return qic.Qic(rc=list(rc), zs=list(zs), nfp=int(N),
                   B0_vals=list(B0v), omn=True, omn_method=omn_method,
                   k_buffer=k_buffer, p_buffer=p_buffer, order=order,
                   d_over_curvature_cvals=list(dcv),
                   nphi=nphi, p2=0.0, I2=0.0, **kw)


# --------------------------------------------------------------------------
# 6. boundary
# --------------------------------------------------------------------------
def boundary_dict(stel, nfp, r, mpol, ntor, ntheta=24, ntor_fit=15):
    import qic.util
    from constellaration.geometry import surface_rz_fourier as srf
    from constellaration.mhd import near_axis_configuration as nac
    R2D, Z2D, _ = stel.Frenet_to_cylindrical(r=r, ntheta=ntheta)
    r_cos, _, _, z_sin = qic.util.to_Fourier(
        R_2D=R2D, Z_2D=Z2D, nfp=nfp, mpol=ntheta // 2, ntor=ntor_fit, lasym=False)
    raw = srf.SurfaceRZFourier(r_cos=r_cos.T, z_sin=z_sin.T,
                               n_field_periods=nfp, is_stellarator_symmetric=True)
    sm = nac.smooth_and_set_max_mode_numbers(
        raw, max_poloidal_mode=mpol, max_toroidal_mode=ntor)
    return {"r_cos": np.asarray(sm.r_cos).tolist(),
            "z_sin": np.asarray(sm.z_sin).tolist(),
            "r_sin": None, "z_cos": None,
            "n_field_periods": int(nfp), "is_stellarator_symmetric": True}


def blend_to_ellipse(bd, lam):
    """Continuation towards the m<=1 (rotating-ellipse) core: keep the m=0/m=1
    rows, damp everything above by lam.  wiki/05's ellipse-blend fallback."""
    out = json.loads(json.dumps(bd))
    for key in ("r_cos", "z_sin"):
        a = np.array(out[key], dtype=float)
        if a.shape[0] > 2:
            a[2:, :] *= lam
        out[key] = a.tolist()
    return out


def diag(stel):
    d = {}
    for name, fn in (("iota", lambda: float(stel.iota)),
                     ("max_elongation", lambda: float(np.max(stel.elongation))),
                     ("qi_consistency_r1",
                      lambda: float(stel.min_geo_qi_consistency(order=1))),
                     ("helicity", lambda: float(stel.helicity)),
                     ("B0_max", lambda: float(np.max(stel.B0))),
                     ("B0_min", lambda: float(np.min(stel.B0))),
                     ("d_max", lambda: float(np.max(np.abs(stel.d)))),
                     ("curvature_max", lambda: float(np.max(np.abs(stel.curvature)))),
                     ("torsion_absmax", lambda: float(np.max(np.abs(stel.torsion))))):
        try:
            d[name] = fn()
        except Exception as e:
            d[name] = f"ERR {e!r}"[:120]
    if isinstance(d.get("B0_max"), float):
        d["mirror_on_axis"] = (d["B0_max"] - d["B0_min"]) / (d["B0_max"] + d["B0_min"])
    return d


def build_one(cand, r, mpol, ntor, omn_method, k_buffer, nphi, blend=None):
    inp = cand["input_features"]
    N = int(cand["db_ref"]["N_field_periods"])
    ax = axis_from_kappa_tau(N, inp["kappa_s1"], inp["kappa_s2"],
                             inp["tau_c0"], inp["tau_c1"], inp["tau_c2"])
    B0v = b0_vals(inp["Delta_mirror"], inp["lambda_B"])
    dcv, dres = dbar_cvals(N, inp["Delta_mirror"], inp["lambda_B"],
                           inp["rho_c0"], inp["rho_c1"], inp["rho_c2"])
    stel = make_stel(N, ax["rc"], ax["zs"], B0v, dcv,
                     omn_method=omn_method, k_buffer=k_buffer, nphi=nphi)
    bd = boundary_dict(stel, N, r, mpol, ntor)
    if blend is not None:
        bd = blend_to_ellipse(bd, blend)
    meta = {
        "candidate": cand["db_ref"], "bucket": cand["bucket"],
        "input_features": inp, "axis": ax,
        "B0_vals": B0v, "d_over_curvature_cvals": dcv, "dbar_fit_resid": dres,
        "r": r, "aspect_target": 1.0 / r, "mpol": mpol, "ntor": ntor,
        "omn_method": omn_method, "k_buffer": k_buffer, "nphi": nphi,
        "blend": blend,
        "near_axis_prediction": cand["p2_proxies"],
        "db_metrics": cand["db_metrics"],
        "pinned_r1_diag": diag(stel),
        "route": "b (pinned qic 0.3.4, axis+B0+rho design transfer; the "
                 "database's own first-order solve is NOT reused)",
    }
    return bd, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS_DIR / "qidb-route-b"))
    ap.add_argument("--candidates", default=CANDIDATES)
    ap.add_argument("--buckets", default="B_geo_and_epseff2pct,B2_geo_and_epseff1pct")
    ap.add_argument("--per-bucket", type=int, default=1)
    ap.add_argument("--radii", default="0.1,0.08,0.06")
    ap.add_argument("--mpol-ntor", default="3:3,5:5")
    ap.add_argument("--omn-method", default="non-zone-smoother")
    ap.add_argument("--k-buffer", type=int, default=2)
    ap.add_argument("--nphi", type=int, default=131)
    ap.add_argument("--blends", default="", help="comma list of lambda for the "
                    "ellipse-blend fallback, e.g. 0.75")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = json.load(open(a.candidates))["candidates"]
    want = [b.strip() for b in a.buckets.split(",") if b.strip()]
    picked = []
    for b in want:
        picked += [c for c in cands if c["bucket"] == b][: a.per_bucket]

    radii = [float(x) for x in a.radii.split(",")]
    mns = [tuple(int(v) for v in p.split(":")) for p in a.mpol_ntor.split(",")]
    blends = [None] + [float(x) for x in a.blends.split(",") if x.strip()]

    index = []
    for c in picked:
        N = c["db_ref"]["N_field_periods"]
        tag = f"{c['bucket']}-N{N}-cfg{c['db_ref']['configurationIndex']}"
        for r in radii:
            for mpol, ntor in mns:
                for bl in blends:
                    name = (f"{tag}-r{r:g}-m{mpol}n{ntor}"
                            + (f"-bl{bl:g}" if bl is not None else ""))
                    rec = {"name": name, "bucket": c["bucket"], "N": N,
                           "r": r, "mpol": mpol, "ntor": ntor, "blend": bl}
                    try:
                        bd, meta = build_one(c, r, mpol, ntor, a.omn_method,
                                             a.k_buffer, a.nphi, blend=bl)
                        (out / f"{name}.json").write_text(
                            json.dumps({"boundary": bd, "meta": meta}, indent=1))
                        rec["ok"] = True
                        rec["diag"] = meta["pinned_r1_diag"]
                        rec["axis"] = {k: meta["axis"][k] for k in
                                       ("closure_err", "asymmetry", "helical_amp",
                                        "R_min_over_R0")}
                    except Exception as e:
                        rec["ok"] = False
                        rec["error"] = f"{e!r}"[:300]
                        rec["trace"] = traceback.format_exc()[-500:]
                    index.append(rec)
                    print(json.dumps({k: v for k, v in rec.items()
                                      if k != "trace"}), flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    ok = sum(1 for r in index if r.get("ok"))
    print(f"BUILT {ok}/{len(index)} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
