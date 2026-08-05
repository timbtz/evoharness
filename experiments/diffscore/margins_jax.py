"""Deliverable A: differentiable P2 constraint margins (constellaration 0.2.6).

Value-parity targets vs the pinned oracle (see difftest): aspect exact
(boundary-only, VMEC definition reproduced); elongation via elongation_jax
(boundary + axis inputs); edge_iota and mirror from wout arrays (exact array
math); normalized violations reproducing problems.py:240-264 verbatim.

Public entry points:
  aspect_ratio(r_cos, z_sin, nfp)                      boundary-only, exact
  aspect_grad(r_cos, z_sin, nfp)                       d(aspect)/d(coeffs)
  edge_iota_over_nfp(iotaf, nfp)
  edge_mirror_ratio(bmnc, s_half, s_full, xm_nyq, xn_nyq, mpol, ntor, nfp)
  max_elongation_wout(...)                             see elongation_jax
  normalized_violations(aspect, edge_iota_nfp, qi, mirror, elongation)
  parity_report(rows, ...)                             difftest hook
"""

import json

import jax
import jax.numpy as jnp
import numpy as np

from experiments.diffscore import elongation_jax

jax.config.update("jax_enable_x64", True)

# problems.py:214-224 bounds, in violation order
BOUNDS = jnp.array([10.0, 0.25, -4.0, 0.2, 5.0])


def _boundary_grids(r_cos, n_theta=128, n_phi=128):
    mpol, two_ntor1 = r_cos.shape
    ntor = (two_ntor1 - 1) // 2
    m = jnp.arange(mpol)[:, None]
    n = jnp.arange(-ntor, ntor + 1)[None, :]
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    return m, n, theta


def aspect_ratio(r_cos, z_sin, nfp, n_theta=128, n_phi=128):
    """VMEC aspect ratio from the boundary alone (verified 7e-15 vs
    wout.aspect): Aminor = sqrt(<area(phi)>/pi), Rmajor = V/(2 pi^2 Aminor^2),
    aspect = Rmajor/Aminor. Trapezoid sums are exact for the band-limited
    boundary."""
    r_cos, z_sin = jnp.asarray(r_cos), jnp.asarray(z_sin)
    m, n, theta = _boundary_grids(r_cos, n_theta, n_phi)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, n_phi, endpoint=False)
    ang = (m[..., None, None] * theta[None, None, :, None]
           - nfp * n[..., None, None] * phi[None, None, None, :])
    cos_a, sin_a = jnp.cos(ang), jnp.sin(ang)
    R = jnp.sum(r_cos[..., None, None] * cos_a, axis=(0, 1))
    dRdt = jnp.sum(-m[..., None, None] * r_cos[..., None, None] * sin_a,
                   axis=(0, 1))
    Z = jnp.sum(z_sin[..., None, None] * sin_a, axis=(0, 1))
    dZdt = jnp.sum(m[..., None, None] * z_sin[..., None, None] * cos_a,
                   axis=(0, 1))
    dtheta = 2.0 * jnp.pi / n_theta
    area = jnp.abs(-jnp.sum(Z * dRdt, axis=0) * dtheta)      # per phi
    mean_area = jnp.mean(area)
    volume = jnp.abs(jnp.mean(jnp.sum(0.5 * R**2 * dZdt, axis=0) * dtheta)
                     * 2.0 * jnp.pi)
    aminor = jnp.sqrt(mean_area / jnp.pi)
    rmajor = volume / (2.0 * jnp.pi**2 * aminor**2)
    return rmajor / aminor


aspect_grad = jax.jit(jax.grad(aspect_ratio, argnums=(0, 1)),
                      static_argnums=(2,))


def aspect_from_boundary(boundary: dict):
    """Aspect from an archive/seed-bank boundary dict; rejects
    non-stellarator-symmetric input like the pinned code does."""
    if boundary.get("r_sin") is not None or boundary.get("z_cos") is not None \
            or not boundary.get("is_stellarator_symmetric", True):
        raise NotImplementedError(
            "Non-stellarator-symmetric boundaries not supported.")
    return aspect_ratio(np.array(boundary["r_cos"]),
                        np.array(boundary["z_sin"]),
                        boundary["n_field_periods"])


def edge_iota_over_nfp(iotaf, nfp):
    """forward_model.py:135-146 evaluates the iotaf profile at rho=1, which is
    the last full-mesh knot of an interpolating spline -> exactly iotaf[-1]."""
    return iotaf[-1] / nfp


def edge_mirror_ratio(bmnc, s_half, s_full, xm_nyq, xn_nyq, mpol, ntor, nfp):
    """magnetics_utils.magnetic_mirror_ratio at rho=1: |B| Fourier coefficients
    linearly extrapolated from the half mesh to s=1 (interp1d
    fill_value='extrapolate'), evaluated on the Nyquist grid, then
    (max-min)/(max+min). Profile evaluation at rho=1 hits the last knot ->
    exact."""
    b = bmnc.T[1:, :]  # (n_half_surfaces, n_modes), dropping the empty s=0 row
    x = s_half[1:]
    t = (1.0 - x[-2]) / (x[-1] - x[-2])
    b_edge = b[-2] + t * (b[-1] - b[-2])
    n_theta = 2 * mpol + 6
    n_phi = 2 * ntor + 4
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp / 2.0, n_phi)
    ang = (xm_nyq[:, None, None] * theta[None, :, None]
           - xn_nyq[:, None, None] * phi[None, None, :])
    B = jnp.sum(b_edge[:, None, None] * jnp.cos(ang), axis=0)
    bmax, bmin = jnp.max(B), jnp.min(B)
    return (bmax - bmin) / (bmax + bmin)


def max_elongation_wout(rmnc, zmns, xm, xn, raxis_cc, zaxis_cs, mpol, ntor,
                        nfp):
    """Elongation with the reference's grid sizes derived from wout mpol/ntor
    (forward_model.py:113-125)."""
    n_theta = 2 * mpol + 6
    n_phi = 2 * ntor + 4
    return elongation_jax.max_elongation(
        rmnc.T[-1, :], zmns.T[-1, :], xm, xn, raxis_cc, zaxis_cs,
        n_theta, n_phi, nfp)


def max_elongation_boundary(r_cos, z_sin, nfp, raxis_cc, zaxis_cs,
                            mpol_w: int, ntor_w: int):
    """Elongation evaluated on the INPUT boundary parameterization (axis still
    from the wout). Same geometric surface as max_elongation_wout but the
    theta samples land on different physical points (vmecpp reparameterizes
    theta), so values differ at discretization level — use for boundary-space
    gradients/walk directions, not for oracle parity."""
    r_cos, z_sin = jnp.asarray(r_cos), jnp.asarray(z_sin)
    mpol_b, two = r_cos.shape
    ntor_b = (two - 1) // 2
    mm, nn = np.meshgrid(np.arange(mpol_b), np.arange(-ntor_b, ntor_b + 1),
                         indexing="ij")
    xm = jnp.asarray(mm.ravel(), float)
    xn = jnp.asarray(-nfp * nn.ravel(), float)
    xn_axis = nfp * np.arange(np.asarray(raxis_cc).shape[0], dtype=float)
    # vmecpp's wout parameterization is the input surface mirrored in Z
    # (measured: wout(phi) = input(-phi)); the wout axis must be mirrored
    # back (zaxis -> -zaxis) to be consistent with the input surface
    return elongation_jax.max_elongation(
        r_cos.ravel(), z_sin.ravel(), xm, xn, jnp.asarray(raxis_cc),
        -jnp.asarray(zaxis_cs), 2 * mpol_w + 6, 2 * ntor_w + 4, nfp,
        xn_axis=xn_axis)


def normalized_violations(aspect, edge_iota_nfp, qi, mirror, elongation):
    """problems.py:240-264 verbatim (order and |target| normalization)."""
    v = jnp.array([aspect - BOUNDS[0],
                   BOUNDS[1] - jnp.abs(edge_iota_nfp),
                   jnp.log10(qi) - BOUNDS[2],
                   mirror - BOUNDS[3],
                   elongation - BOUNDS[4]])
    return v / jnp.abs(BOUNDS)


def feasibility(violations):
    return jnp.max(jnp.maximum(violations, 0.0))


# ---------------------------------------------------------------------------
# difftest hooks
# ---------------------------------------------------------------------------

def _fd_check(f, x, g, steps=(1e-6, 1e-5), n_dirs=3, seed=0):
    """Max relative error of jax.grad vs central finite differences along
    random directions."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_dirs):
        d = rng.standard_normal(x.shape)
        d /= np.linalg.norm(d)
        g_dir = float(np.sum(np.asarray(g) * d))
        best = np.inf
        for h in steps:
            fd = (float(f(x + h * d)) - float(f(x - h * d))) / (2 * h)
            rel = abs(fd - g_dir) / max(abs(fd), abs(g_dir), 1e-300)
            best = min(best, rel)
        worst = max(worst, best)
    return worst


def parity_report(rows, grad_check_every=0):
    from experiments.diffscore import lgradb_jax
    from experiments.diffscore.difftest import norms
    deltas = {"aspect": [], "edge_iota": [], "mirror": [], "elongation": [],
              "lgradb": [], "violations": [], "feasibility": []}
    grad_errs = {"aspect": [], "elongation": []}
    for i, r in enumerate(rows):
        if i % 10 == 0:
            import gc
            jax.clear_caches()  # per-row closures accumulate compile caches
            gc.collect()
            print(f"margins parity {i}/{len(rows)}", flush=True)
        z = np.load(r["npz"])
        bd = r["boundary"]
        nfp = int(z["nfp"])
        a = float(aspect_ratio(np.array(bd["r_cos"]), np.array(bd["z_sin"]),
                               bd["n_field_periods"]))
        ei = float(edge_iota_over_nfp(z["iotaf"], nfp))
        mi = float(edge_mirror_ratio(
            z["bmnc"], z["normalized_toroidal_flux_half_grid_mesh"],
            z["normalized_toroidal_flux_full_grid_mesh"], z["xm_nyq"],
            z["xn_nyq"], int(z["mpol"]), int(z["ntor"]), nfp))
        el, _ = max_elongation_wout(z["rmnc"], z["zmns"], z["xm"], z["xn"],
                                    z["raxis_cc"], z["zaxis_cs"],
                                    int(z["mpol"]), int(z["ntor"]), nfp)
        el = float(el)
        lg = float(lgradb_jax.min_normalized_l_grad_b(
            rmnc=z["rmnc"], zmns=z["zmns"], gmnc=z["gmnc"], bmnc=z["bmnc"],
            bsupumnc=z["bsupumnc"], bsupvmnc=z["bsupvmnc"], xm=z["xm"],
            xn=z["xn"], xm_nyq=z["xm_nyq"], xn_nyq=z["xn_nyq"],
            ns=int(z["ns"]), aminor_p=float(z["Aminor_p"]),
            mpol=int(z["mpol"]), ntor=int(z["ntor"]), nfp=nfp,
            lasym=bool(z["lasym"])))
        deltas["lgradb"].append(
            lg - r["minimum_normalized_magnetic_gradient_scale_length"])
        deltas["aspect"].append(a - r["aspect_ratio"])
        deltas["edge_iota"].append(
            ei - r["edge_rotational_transform_over_n_field_periods"])
        deltas["mirror"].append(mi - r["edge_magnetic_mirror_ratio"])
        deltas["elongation"].append(el - r["max_elongation"])
        v = np.asarray(normalized_violations(a, ei, r["qi"], mi, el))
        deltas["violations"].extend(v - np.array(r["violations"]))
        deltas["feasibility"].append(
            float(feasibility(jnp.asarray(v))) - r["feasibility"])
        if grad_check_every and i % grad_check_every == 0:
            rc = np.array(bd["r_cos"])
            zs = np.array(bd["z_sin"])
            g_rc, _ = aspect_grad(rc, zs, bd["n_field_periods"])
            grad_errs["aspect"].append(_fd_check(
                lambda x: aspect_ratio(x, zs, bd["n_field_periods"]), rc,
                g_rc))
            rl = np.asarray(z["rmnc"]).T[-1, :].copy()

            def elong_of_rl(x, z=z, nfp=nfp):
                v, _ = elongation_jax.max_elongation(
                    x, np.asarray(z["zmns"]).T[-1, :], z["xm"], z["xn"],
                    z["raxis_cc"], z["zaxis_cs"], 2 * int(z["mpol"]) + 6,
                    2 * int(z["ntor"]) + 4, nfp)
                return v
            g_el = jax.grad(elong_of_rl)(rl)
            grad_errs["elongation"].append(_fd_check(elong_of_rl, rl, g_el))
    out = {k: norms(np.array(v)) for k, v in deltas.items()}
    for k, v in grad_errs.items():
        if v:
            out[f"grad_fd_relerr_{k}"] = {"max": float(np.max(v)),
                                          "median": float(np.median(v))}
    print(json.dumps(out, indent=2))
    return out
