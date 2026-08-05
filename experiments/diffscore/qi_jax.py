"""Deliverable B: JAX port of the scored QI residual (omnigeneity/qi.py,
constellaration 0.2.6; Goodman et al. JPP 2023).

Inputs are Boozer arrays (Option 1 of the plan): bmnc_b, xm_b, xn_b, iota on
the input flux grid, nfp — differentiate w.r.t. bmnc_b (and iota). Structure
mirrors qi.py stage-by-stage so difftest can compare each stage.

Non-smooth reference ops and their treatment:
  - argmin/argmax well centering: hard indices (a.e.-correct subgradients,
    maxpool-style). `smoothing` does not soften these (documented).
  - minimum.accumulate squash: exact cummin at smoothing=0; smooth running
    min via -tau*logcumsumexp(-y/tau) for smoothing=tau>0.
  - global min/max normalization: exact at smoothing=0, softmin/softmax at tau.
  - bounce crossings: index selection is hard; the crossing location itself is
    linear interpolation (differentiable in the values).
  - the monotonicity repair loop fires on the flattened bounce-point array
    exactly as the reference's aliased numpy views do.
The weight "spline" (UnivariateSpline, default k=3, s=len(x)) degenerates to a
cubic least-squares polynomial; reproduced as an exact linear functional.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_EPS = float(np.finfo(float).eps)


class QIConfig:
    def __init__(self, n_field_lines=75, n_toroidal_points=601,
                 n_field_contours=401, n_output_toroidal_points=2000):
        self.nalpha = n_field_lines
        self.nphi = n_toroidal_points
        self.nBj = n_field_contours
        self.nphi_out = n_output_toroidal_points


def _modb_grid(bmnc_b_js, xm_b, xn_b, iota_js, phi_start, nfp, cfg,
               smoothing):
    """|B| on the (nphi, nalpha) grid, normalized to [0,1].

    theta(phi, alpha) = t0(alpha) + iota*phi, so the reference's
    (n_modes, nphi, nalpha) angle tensor factors exactly:
    cos(m*t0 + (m*iota - n)*phi) = cos(kappa*phi)cos(m*t0)
                                 - sin(kappa*phi)sin(m*t0),
    turning the mode sum into two small matmuls (memory: the eager 3-D
    tensor OOMed the shared box)."""
    phi_end = phi_start + 2.0 * np.pi / nfp
    phis = jnp.linspace(phi_start, phi_end, cfg.nphi)
    theta_start = -iota_js * phi_start
    t0 = jnp.linspace(theta_start, theta_start + 2.0 * jnp.pi, cfg.nalpha)
    kappa = xm_b * iota_js - xn_b                        # (n_modes,)
    kphi = kappa[:, None] * phis[None, :]                # (n_modes, nphi)
    mt0 = xm_b[:, None] * t0[None, :]                    # (n_modes, nalpha)
    modb = (jnp.cos(kphi).T @ (bmnc_b_js[:, None] * jnp.cos(mt0))
            - jnp.sin(kphi).T @ (bmnc_b_js[:, None] * jnp.sin(mt0)))
    if smoothing > 0.0:
        t = smoothing
        lo = -t * jax.nn.logsumexp(-modb / t)
        hi = t * jax.nn.logsumexp(modb / t)
    else:
        lo, hi = jnp.min(modb), jnp.max(modb)
    return phis, (modb - lo) / (hi - lo)


def _cummin(y, smoothing):
    if smoothing > 0.0:
        t = smoothing
        return -t * jax.lax.cumlogsumexp(-y / t)
    return jax.lax.associative_scan(jnp.minimum, y)


def squash_stretch_line(modb_line, smoothing=0.0):
    """SQUASH + STRETCH of one field line; returns the combined
    monotone-down-then-up profile (same length), exactly as the reference
    concatenation of left[:-1] and right."""
    n = modb_line.shape[0]
    idx = jnp.arange(n)
    mi = jnp.argmin(modb_line)

    # left side: indices 0..mi; plateau before argmax(left), then cummin
    left_mask = idx <= mi
    left_vals = jnp.where(left_mask, modb_line, -jnp.inf)
    la = jnp.argmax(left_vals)
    lv = modb_line[la]
    pre = jnp.where(idx < la, lv, jnp.where(left_mask, modb_line, jnp.inf))
    left_sq = _cummin(pre, smoothing)
    y_mi = left_sq[mi]
    left_str = (left_sq - y_mi) / (left_sq[0] - y_mi)

    # right side: indices mi..n-1; plateau after argmax(right), reversed cummin
    right_mask = idx >= mi
    right_vals = jnp.where(right_mask, modb_line, -jnp.inf)
    ra = jnp.argmax(right_vals)
    rv = modb_line[ra]
    post = jnp.where(idx > ra, rv, jnp.where(right_mask, modb_line, jnp.inf))
    right_sq = jnp.flip(_cummin(jnp.flip(post), smoothing))
    ry_mi = right_sq[mi]
    right_str = (right_sq - ry_mi) / (right_sq[n - 1] - ry_mi)

    return jnp.where(idx < mi, left_str, right_str)


def fieldline_weight(phis, diff2):
    """(phi_end-phi_start) / integral of the degenerate smoothing spline =
    cubic LSQ polynomial fit of diff2 over phis."""
    t = phis - phis[phis.shape[0] // 2]
    V = jnp.stack([jnp.ones_like(t), t, t**2, t**3], axis=-1)
    coef, _, _, _ = jnp.linalg.lstsq(V, diff2)
    a, b = t[0], t[-1]
    powers = jnp.array([(b - a),
                        (b**2 - a**2) / 2.0,
                        (b**3 - a**3) / 3.0,
                        (b**4 - a**4) / 4.0])
    integral = jnp.dot(coef, powers)
    return (phis[-1] - phis[0]) / integral


def bounce_points_line(phis, y, Bjs):
    """Reference _find_bounce_points for every level; returns (nBj, 2) of
    (phi_left, phi_right)."""
    phis, y, Bjs = jnp.asarray(phis), jnp.asarray(y), jnp.asarray(Bjs)
    n = y.shape[0]
    d = y - Bjs[:, None]                              # (nBj, n)
    prod = d[:, :-1] * d[:, 1:]
    crossing = prod < 0.0                             # (nBj, n-1)
    first = jnp.argmax(crossing, axis=1)
    last = n - 2 - jnp.argmax(jnp.flip(crossing, axis=1), axis=1)

    def lin(ci, Bj):
        dy = y[ci] - y[ci + 1]
        dx = phis[ci] - phis[ci + 1]
        m = dy / dx
        b = y[ci] - m * phis[ci]
        # safe divide: on plateaus m == 0 and the untaken branch would leak
        # NaN through the where in reverse mode
        ok = m != 0.0
        m_safe = jnp.where(ok, m, 1.0)
        return jnp.where(ok, (Bj - b) / m_safe, phis[ci])

    left = jax.vmap(lin)(first, Bjs)
    right = jax.vmap(lin)(last, Bjs)

    phi_min = phis[jnp.argmin(y)]
    lo_case = Bjs - 0.0 < _EPS                        # includes Bj < min
    hi_case = 1.0 - Bjs < _EPS
    left = jnp.where(lo_case, phi_min, jnp.where(hi_case, phis[0], left))
    right = jnp.where(lo_case, phi_min, jnp.where(hi_case, phis[-1], right))
    return jnp.stack([left, right], axis=-1)


def _repair(x, nBj):
    """The reference's aliased-view monotonicity fix-up on the flat
    (2*nBj-1,) bounce-point array; runs only when x is not strictly
    increasing (the reference repairs only after the spline ctor raises)."""
    def fix(x):
        def body(il, x):
            L1, L0 = x[il + 1], x[il]
            ri = 2 * nBj - il - 3                     # R[-il-2]
            def bad_left(x):
                x = x.at[ri].add(L0 - L1 + 1e-12)
                return x.at[il + 1].set(L0 + 1e-12)
            x = jax.lax.cond(L1 - L0 < 0.0, bad_left, lambda x: x, x)
            def bad_right(x):
                x = x.at[il + 1].add(x[ri + 1] - x[ri] - 1e-12)
                return x.at[ri].set(x[ri + 1] - 1e-12)
            x = jax.lax.cond(x[ri + 1] - x[ri] < 0.0, bad_right,
                             lambda x: x, x)
            return x
        return jax.lax.fori_loop(0, nBj - 1, body, x)
    monotone = jnp.all(x[1:] > x[:-1])
    return jax.lax.cond(monotone, lambda x: x, fix, x)


def _interp_extrap(xq, x, y):
    """k=1 s=0 UnivariateSpline evaluation incl. linear end extrapolation."""
    yq = jnp.interp(xq, x, y)
    lo = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (xq - x[0])
    hi = y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (xq - x[-1])
    return jnp.where(xq < x[0], lo, jnp.where(xq > x[-1], hi, yq))


from functools import partial


@partial(jax.jit, static_argnums=(4, 5, 6))
def _one_surface(bmnc_js, xm_b, xn_b, iota_js_and_phi_start, nfp, cfg_dims,
                 smoothing):
    cfg = QIConfig(*cfg_dims)
    nBj = cfg.nBj
    iota_js, phi_start = iota_js_and_phi_start
    Bjs = jnp.linspace(0.0, 1.0, nBj)
    qi_modb = jnp.concatenate([jnp.flip(Bjs), Bjs[1:]])

    def one_surface(bmnc_js, iota_js):
        phis, modb = _modb_grid(bmnc_js, xm_b, xn_b, iota_js, phi_start, nfp,
                                cfg, smoothing)
        phi_end = phis[-1]
        sq = jax.vmap(lambda c: squash_stretch_line(c, smoothing),
                      in_axes=1, out_axes=1)(modb)     # (nphi, nalpha)
        diff2 = (modb - sq) ** 2
        weights = jax.vmap(lambda c: fieldline_weight(phis, c),
                           in_axes=1)(diff2)           # (nalpha,)
        weights = weights / jnp.sum(weights)
        # sequential over field lines: the (nBj, nphi) crossing temporaries
        # are ~2 MB each; vmapping all 75 lines at once OOMs the shared box
        bp = jax.lax.map(lambda c: bounce_points_line(phis, c, Bjs),
                         sq.T)                          # (nalpha, nBj, 2)
        distances = bp[..., 1] - bp[..., 0]             # (nalpha, nBj)
        mean_distances = jnp.sum(distances * weights[:, None], axis=0)

        # phi_bounce_points: left at nBj-j-1, right at nBj+j-1
        pbp = jnp.concatenate([jnp.flip(bp[..., 0], axis=1),
                               bp[:, 1:, 1]], axis=1)   # (nalpha, 2nBj-1)
        out_phi = jnp.linspace(phi_start, phi_end, cfg.nphi_out)

        def one_alpha(args):
            pbp_a, dist_a, modb_a = args
            delta = (dist_a - mean_distances) / 2.0
            left = pbp_a[:nBj] + jnp.flip(delta)
            right = pbp_a[nBj - 1:] - delta
            x = jnp.concatenate([left[:nBj - 1], right])
            x = _repair(x, nBj)
            shuffled = _interp_extrap(out_phi, x, qi_modb)
            orig = jnp.interp(out_phi, phis, modb_a)
            return (shuffled - orig) / jnp.sqrt(1.0 * cfg.nphi_out)

        pen = jax.lax.map(one_alpha, (pbp, distances, modb.T))
        fl_norm = jnp.sum(jnp.full(cfg.nalpha, 1.0 / cfg.nalpha))
        return pen * fl_norm

    return one_surface(bmnc_js, iota_js)


def qi_residual(bmnc_b, xm_b, xn_b, iota_in, s_in, s_b, nfp, bmnc_11,
                cfg=None, smoothing=0.0):
    """Full port of quasi_isodynamicity_residual; returns residuals of shape
    (n_surfaces, nalpha, nphi_out). qi metric = sum(residuals**2)."""
    cfg = cfg or QIConfig()
    cfg_dims = (cfg.nalpha, cfg.nphi, cfg.nBj, cfg.nphi_out)
    phi_start = np.pi / nfp if float(bmnc_11) < 0 else 0.0
    # k=1 s=0 spline incl. linear extrapolation: s_b=1.0 can lie beyond the
    # last input surface, where jnp.interp would clamp
    iota_b = _interp_extrap(jnp.asarray(s_b), jnp.asarray(s_in),
                            jnp.asarray(iota_in))
    bmnc_b = jnp.asarray(bmnc_b)
    res = jnp.stack([
        _one_surface(bmnc_b[:, js], jnp.asarray(xm_b), jnp.asarray(xn_b),
                     (iota_b[js], phi_start), int(nfp), cfg_dims,
                     float(smoothing))
        for js in range(bmnc_b.shape[1])])
    return res / jnp.sqrt(1.0 * cfg.nalpha)


def qi_metric(bmnc_b, xm_b, xn_b, iota_in, s_in, s_b, nfp, bmnc_11,
              cfg=None, smoothing=0.0):
    r = qi_residual(bmnc_b, xm_b, xn_b, iota_in, s_in, s_b, nfp, bmnc_11,
                    cfg, smoothing)
    return jnp.sum(r**2)


# ---------------------------------------------------------------------------
# difftest hook
# ---------------------------------------------------------------------------

def _reference_boozer_stub(z):
    """Duck-typed stand-in for BoozerOutput feeding the pinned qi.py."""
    import types
    bmnc = np.zeros((2, 2))
    bmnc[1, 1] = float(z["bmnc_11"])
    return types.SimpleNamespace(
        n_boozer_flux_surfaces=int(np.asarray(z["s_b"]).size),
        n_field_periods=int(z["nfp"]), bmnc=bmnc,
        xm_b=np.asarray(z["xm_b"]), xn_b=np.asarray(z["xn_b"]),
        normalized_toroidal_flux=np.asarray(z["s_in"]),
        boozer_normalized_toroidal_flux=np.atleast_1d(np.asarray(z["s_b"])),
        iota=np.asarray(z["boozer_iota"]), bmnc_b=np.asarray(z["bmnc_b"]))


def _reference_qi(z) -> float:
    """Pinned qi.py end-to-end on cached boozer arrays (raises ValueError
    where the reference itself would)."""
    from constellaration.omnigeneity import qi as qi_ref
    ref = qi_ref.quasi_isodynamicity_residual(
        boozer=_reference_boozer_stub(z), settings=qi_ref.QISettings())
    return float(np.sum(ref.residuals**2))


def parity_report(rows, smoothing=0.0, grad_check_every=0):
    """End-to-end |delta log10 qi| vs the cached pinned-oracle qi (which IS
    the reference: same qi.py on the same boozer arrays — verified bit-exact
    via the stub during development). No in-report reference re-run: it
    materializes multi-GB mode tensors."""
    import json as _json
    from experiments.diffscore.difftest import norms
    d_log10, grad_relerr, skipped = [], [], 0
    for i, r in enumerate(rows):
        if r.get("qi") is None:
            skipped += 1
            continue
        if i % 10 == 0:
            import gc
            jax.clear_caches()
            gc.collect()
            print(f"qi parity {i}/{len(rows)}", flush=True)
        z = np.load(r["npz"])
        args = (z["bmnc_b"], z["xm_b"], z["xn_b"], z["boozer_iota"],
                z["s_in"], np.atleast_1d(z["s_b"]), int(z["nfp"]),
                float(z["bmnc_11"]))
        res = np.asarray(qi_residual(*args, smoothing=smoothing))
        jax_q = float(np.sum(res**2))
        d_log10.append(np.log10(jax_q) - np.log10(r["qi"]))
        if grad_check_every and i % grad_check_every == 0:
            g = jax.grad(lambda b: qi_metric(
                b, *args[1:], smoothing=smoothing))(jnp.asarray(z["bmnc_b"]))
            rng = np.random.default_rng(i)
            dvec = rng.standard_normal(z["bmnc_b"].shape)
            dvec /= np.linalg.norm(dvec)
            gd = float(jnp.sum(g * dvec))
            h = 1e-6 * float(np.max(np.abs(z["bmnc_b"])))
            fp = float(qi_metric(z["bmnc_b"] + h * dvec, *args[1:],
                                 smoothing=smoothing))
            fm = float(qi_metric(z["bmnc_b"] - h * dvec, *args[1:],
                                 smoothing=smoothing))
            fd = (fp - fm) / (2 * h)
            grad_relerr.append(abs(fd - gd) / max(abs(fd), abs(gd), 1e-300))
    out = {"delta_log10_qi": norms(np.array(d_log10)),
           "skipped": skipped, "smoothing": smoothing}
    if grad_relerr:
        out["grad_fd_relerr"] = {"max": float(np.max(grad_relerr)),
                                 "median": float(np.median(grad_relerr))}
    print(_json.dumps(out, indent=2))
    return out
