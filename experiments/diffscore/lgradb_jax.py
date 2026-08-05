"""JAX port of magnetics_utils.normalized_magnetic_gradient_scale_length
(constellaration 0.2.6) — the P2 objective. Differentiable w.r.t. every wout
array input; d(wout)/d(boundary) is Plan-3 (adjoint) territory.

The P2 score is min(L_gradB over the Nyquist grid) * nfp / 20 (problems.py
_score with _normalize_between_bounds(0, 20), forward_model.py:174-181).
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _d_ds_full(x, ds):
    return (11 / 6 * x[-1] - 3 * x[-2] + 3 / 2 * x[-3] - 1 / 3 * x[-4]) / ds


def _d_ds_half(x, ds):
    d_n = (1.5 * x[-1] - 2.0 * x[-2] + 0.5 * x[-3]) / ds
    d_n1 = (x[-1] - x[-3]) / (2 * ds)
    d_n2 = (x[-2] - x[-4]) / (2 * ds)
    return 7 / 4 * d_n - d_n1 + 1 / 4 * d_n2


def _at_boundary_half(x):
    return 7 / 4 * x[-1] - x[-2] + 1 / 4 * x[-3]


def min_normalized_l_grad_b(rmnc, zmns, gmnc, bmnc, bsupumnc, bsupvmnc,
                            xm, xn, xm_nyq, xn_nyq, ns: int, aminor_p,
                            mpol: int, ntor: int, nfp: int, lasym: bool):
    """min over the Nyquist theta-phi grid of L_gradB / Aminor_p, times nfp
    (i.e. metrics.minimum_normalized_magnetic_gradient_scale_length)."""
    n_theta = 2 * mpol + 6
    n_phi = 2 * ntor + 4
    phi_ub = 2.0 * np.pi / nfp / (1 + int(not lasym))
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta)
    phi = jnp.linspace(0.0, phi_ub, n_phi)
    t2 = theta[:, None] + 0.0 * phi[None, :]
    p2 = 0.0 * theta[:, None] + phi[None, :]

    rmnc, zmns = rmnc.T, zmns.T
    gmnc, bmnc = gmnc.T[1:, :], bmnc.T[1:, :]
    bsupumnc, bsupvmnc = bsupumnc.T[1:, :], bsupvmnc.T[1:, :]

    s_full = np.linspace(0.0, 1.0, ns)
    ds = s_full[2] - s_full[1]

    d_rmnc = _d_ds_full(rmnc, ds)
    d_zmns = _d_ds_full(zmns, ds)
    d_bmnc = _d_ds_half(bmnc, ds)
    d_bsupu = _d_ds_half(bsupumnc, ds)
    d_bsupv = _d_ds_half(bsupvmnc, ds)

    rb, zb = rmnc[-1], zmns[-1]
    gb, bb = _at_boundary_half(gmnc), _at_boundary_half(bmnc)
    bub, bvb = _at_boundary_half(bsupumnc), _at_boundary_half(bsupvmnc)

    def synth(coef, modes_m, modes_n, basis):
        ang = (modes_m[:, None, None] * t2[None] -
               modes_n[:, None, None] * p2[None])
        f = jnp.cos(ang) if basis == "cos" else jnp.sin(ang)
        return jnp.sum(coef[:, None, None] * f, axis=0)

    m_, n_ = jnp.asarray(xm, float), jnp.asarray(xn, float)
    mn_, nn_ = jnp.asarray(xm_nyq, float), jnp.asarray(xn_nyq, float)

    R = synth(rb, m_, n_, "cos")
    d_R_dt = synth(-rb * m_, m_, n_, "sin")
    d_R_dp = synth(rb * n_, m_, n_, "sin")
    d2_R_dt2 = synth(-rb * m_ * m_, m_, n_, "cos")
    d2_R_dtdp = synth(rb * m_ * n_, m_, n_, "cos")
    d2_R_dp2 = synth(-rb * n_ * n_, m_, n_, "cos")
    d_Z_dt = synth(zb * m_, m_, n_, "cos")
    d_Z_dp = synth(-zb * n_, m_, n_, "cos")
    d2_Z_dt2 = synth(-zb * m_ * m_, m_, n_, "sin")
    d2_Z_dtdp = synth(zb * m_ * n_, m_, n_, "sin")
    d2_Z_dp2 = synth(-zb * n_ * n_, m_, n_, "sin")
    d_R_ds = synth(d_rmnc, m_, n_, "cos")
    d_Z_ds = synth(d_zmns, m_, n_, "sin")
    d2_R_dsdt = synth(-d_rmnc * m_, m_, n_, "sin")
    d2_R_dsdp = synth(d_rmnc * n_, m_, n_, "sin")
    d2_Z_dsdt = synth(d_zmns * m_, m_, n_, "cos")
    d2_Z_dsdp = synth(-d_zmns * n_, m_, n_, "cos")

    B = synth(bb, mn_, nn_, "cos")
    sqrt_g = synth(gb, mn_, nn_, "cos")
    Bsu = synth(bub, mn_, nn_, "cos")
    Bsv = synth(bvb, mn_, nn_, "cos")
    d_Bsu_dt = synth(-bub * mn_, mn_, nn_, "sin")
    d_Bsv_dt = synth(-bvb * mn_, mn_, nn_, "sin")
    d_Bsu_dp = synth(bub * nn_, mn_, nn_, "sin")
    d_Bsv_dp = synth(bvb * nn_, mn_, nn_, "sin")
    d_Bsu_ds = synth(d_bsupu, mn_, nn_, "cos")
    d_Bsv_ds = synth(d_bsupv, mn_, nn_, "cos")

    cp, sp = jnp.cos(p2), jnp.sin(p2)

    grad_s = (-d_Z_dt * R / sqrt_g,
              (d_R_dp * d_Z_dt - d_R_dt * d_Z_dp) / sqrt_g,
              d_R_dt * R / sqrt_g)
    grad_t = (d_Z_ds * R / sqrt_g,
              (d_R_ds * d_Z_dp - d_R_dp * d_Z_ds) / sqrt_g,
              -d_R_ds * R / sqrt_g)
    grad_p = (0.0 * sqrt_g, 1.0 / R, 0.0 * sqrt_g)

    def to_xyz(v):
        vR, vphi, vZ = v
        return (vR * cp - vphi * sp, vR * sp + vphi * cp, vZ)

    gsX, gsY, gsZ = to_xyz(grad_s)
    gtX, gtY, gtZ = to_xyz(grad_t)
    gpX, gpY, gpZ = to_xyz(grad_p)

    d_BX_ds = (d_Bsu_ds * d_R_dt * cp + Bsu * d2_R_dsdt * cp
               + d_Bsv_ds * d_R_dp * cp + Bsv * d2_R_dsdp * cp
               - d_Bsv_ds * R * sp - Bsv * d_R_ds * sp)
    d_BX_dt = (d_Bsu_dt * d_R_dt * cp + Bsu * d2_R_dt2 * cp
               + d_Bsv_dt * d_R_dp * cp + Bsv * d2_R_dtdp * cp
               - d_Bsv_dt * R * sp - Bsv * d_R_dt * sp)
    d_BX_dp = (d_Bsu_dp * d_R_dt * cp + Bsu * d2_R_dtdp * cp
               - Bsu * d_R_dt * sp + d_Bsv_dp * d_R_dp * cp
               + Bsv * d2_R_dp2 * cp - Bsv * d_R_dp * sp
               - d_Bsv_dp * R * sp - Bsv * d_R_dp * sp - Bsv * R * cp)
    d_BY_ds = (d_Bsu_ds * d_R_dt * sp + Bsu * d2_R_dsdt * sp
               + d_Bsv_ds * d_R_dp * sp + Bsv * d2_R_dsdp * sp
               + d_Bsv_ds * R * cp + Bsv * d_R_ds * cp)
    d_BY_dt = (d_Bsu_dt * d_R_dt * sp + Bsu * d2_R_dt2 * sp
               + d_Bsv_dt * d_R_dp * sp + Bsv * d2_R_dtdp * sp
               + d_Bsv_dt * R * cp + Bsv * d_R_dt * cp)
    d_BY_dp = (d_Bsu_dp * d_R_dt * sp + Bsu * d2_R_dtdp * sp
               + Bsu * d_R_dt * cp + d_Bsv_dp * d_R_dp * sp
               + Bsv * d2_R_dp2 * sp + Bsv * d_R_dp * cp
               + d_Bsv_dp * R * cp + Bsv * d_R_dp * cp - Bsv * R * sp)
    d_BZ_ds = (d_Bsu_ds * d_Z_dt + Bsu * d2_Z_dsdt
               + d_Bsv_ds * d_Z_dp + Bsv * d2_Z_dsdp)
    d_BZ_dt = (d_Bsu_dt * d_Z_dt + Bsu * d2_Z_dt2
               + d_Bsv_dt * d_Z_dp + Bsv * d2_Z_dtdp)
    d_BZ_dp = (d_Bsu_dp * d_Z_dt + Bsu * d2_Z_dtdp
               + d_Bsv_dp * d_Z_dp + Bsv * d2_Z_dp2)

    gg = 0.0
    comps = [(d_BX_ds, d_BX_dt, d_BX_dp), (d_BY_ds, d_BY_dt, d_BY_dp),
             (d_BZ_ds, d_BZ_dt, d_BZ_dp)]
    for (dds, ddt, ddp) in comps:
        for gX in [(gsX, gtX, gpX), (gsY, gtY, gpY), (gsZ, gtZ, gpZ)]:
            gg = gg + (dds * gX[0] + ddt * gX[1] + ddp * gX[2]) ** 2

    l_grad_b = B * jnp.sqrt(2.0 / gg)
    return jnp.min(l_grad_b / aminor_p) * nfp


def p2_score(min_l_grad_b_normalized):
    """problems.py:233-238: linear 0..20 -> 0..1, clipped."""
    return jnp.clip(min_l_grad_b_normalized / 20.0, 0.0, 1.0)
