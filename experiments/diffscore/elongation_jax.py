"""JAX port of the pinned `geometry_utils.max_elongation` (constellaration 0.2.6).

Faithful to the reference algorithm: cross-sections of the boundary surface in
planes perpendicular to the magnetic-axis tangent, ellipse fitted by perimeter
and area. The two scipy `fsolve` calls become Newton iterations (autodiff
derivatives, warm-started like the reference), `scipy.special.ellipe` becomes
an AGM evaluation. Inputs are wout arrays; the axis (raxis_cc/zaxis_cs) is an
input, not derived from the boundary — d(axis)/d(boundary) is not captured
(measured to move the metric by ~4e-4 between fidelities; see Plan-2 report).
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_PERP_ITERS = 40
_ELLIPSE_ITERS = 40


def ellipe(m):
    """Complete elliptic integral of the 2nd kind via AGM; matches
    scipy.special.ellipe to ~1e-15 for m < 1 (negative m included)."""
    a, b = jnp.ones_like(m), jnp.sqrt(1.0 - m)
    e_sum = 0.5 * m  # 2^{n-1} c_n^2 term for n=0 (c_0^2 = m)
    pow2 = 1.0
    for _ in range(16):
        c = 0.5 * (a - b)
        a, b = 0.5 * (a + b), jnp.sqrt(a * b)
        pow2 *= 2.0
        e_sum = e_sum + pow2 * 0.5 * c**2
    K = jnp.pi / (2.0 * a)
    return K * (1.0 - e_sum)


def _safe_norm(sq):
    """sqrt with zero (not NaN) gradient at exactly-zero inputs (the duplicated
    theta=0/2pi vertex creates zero-length polygon edges)."""
    positive = sq > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, sq, 1.0)), 0.0)


def max_elongation(rmnc_last, zmns_last, xm, xn, raxis_cc, zaxis_cs,
                   n_poloidal_points: int, n_toroidal_points: int,
                   nfp: int, xn_axis=None):
    """Port of geometry_utils.max_elongation; returns (max_elongation,
    elongations_per_phi).

    xn_axis: axis toroidal mode numbers. The reference slices them off the
    surface mode list (`xn[:len(raxis_cc)]`), which only works for wout mode
    ordering (m=0 block first, xn = 0, nfp, 2*nfp, ...). Callers feeding a
    differently-ordered mode list (e.g. the input-boundary parameterization)
    must pass them explicitly."""
    theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, n_poloidal_points)
    phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, n_toroidal_points)

    def boundary_point(theta, phi):
        ang = xm * theta + xn * phi
        rb = jnp.sum(rmnc_last * jnp.cos(ang))
        zb = jnp.sum(zmns_last * jnp.sin(ang))
        return jnp.array([rb * jnp.cos(phi), rb * jnp.sin(phi), zb])

    # axis position and tangent at each phi (verbatim from the reference)
    if xn_axis is None:
        xn_axis = xn[: raxis_cc.shape[0]]
    xn_axis = jnp.asarray(xn_axis)[:, None]
    tor = xn_axis * phi1d[None, :]
    cos_t, sin_t = jnp.cos(tor), jnp.sin(tor)
    R_axis = jnp.sum(raxis_cc[:, None] * cos_t, axis=0)
    Z_axis = jnp.sum(zaxis_cs[:, None] * sin_t, axis=0)
    dR = jnp.sum(-raxis_cc[:, None] * xn_axis * sin_t, axis=0)
    dZ = jnp.sum(zaxis_cs[:, None] * xn_axis * cos_t, axis=0)
    d_l = jnp.sqrt(R_axis**2 + dR**2 + dZ**2)
    tR, tPhi, tZ = dR / d_l, R_axis / d_l, dZ / d_l
    tangents = jnp.stack([tR * jnp.cos(phi1d) - tPhi * jnp.sin(phi1d),
                          tR * jnp.sin(phi1d) + tPhi * jnp.cos(phi1d),
                          tZ], axis=-1)
    axes_xyz = jnp.stack([R_axis * jnp.cos(phi1d), R_axis * jnp.sin(phi1d),
                          Z_axis], axis=-1)

    def perp_phi(theta, phi_init, tangent, axis_xyz):
        def g(phi):
            return jnp.dot(tangent, boundary_point(theta, phi) - axis_xyz)
        dg = jax.grad(g)

        def body(_, phi):
            d = dg(phi)
            step = jnp.where(jnp.abs(d) > 1e-300, g(phi) / d, 0.0)
            return phi - step
        return jax.lax.fori_loop(0, _PERP_ITERS, body, phi_init)

    def cross_section(phi_init, tangent, axis_xyz):
        phis_perp = jax.vmap(
            lambda th: perp_phi(th, phi_init, tangent, axis_xyz))(theta1d)
        pts = jax.vmap(boundary_point)(theta1d, phis_perp)
        pts = pts - jnp.outer(pts @ tangent, tangent)
        return pts  # (n_theta, 3)

    def polygon_perimeter_area(pts):
        d = pts - jnp.roll(pts, 1, axis=0)
        perimeter = jnp.sum(_safe_norm(jnp.sum(d**2, axis=-1)))
        nxt = jnp.roll(pts, -1, axis=0)
        total = jnp.sum(jnp.cross(pts, nxt), axis=0)
        a, b, c = pts[0], pts[1], pts[2]
        # determinant-based plane normal, as in the reference
        nx = jnp.linalg.det(jnp.stack([
            jnp.array([1.0, a[1], a[2]]), jnp.array([1.0, b[1], b[2]]),
            jnp.array([1.0, c[1], c[2]])]))
        ny = jnp.linalg.det(jnp.stack([
            jnp.array([a[0], 1.0, a[2]]), jnp.array([b[0], 1.0, b[2]]),
            jnp.array([c[0], 1.0, c[2]])]))
        nz = jnp.linalg.det(jnp.stack([
            jnp.array([a[0], a[1], 1.0]), jnp.array([b[0], b[1], 1.0]),
            jnp.array([c[0], c[1], 1.0])]))
        normal = jnp.array([nx, ny, nz])
        normal = normal / jnp.sqrt(jnp.sum(normal**2))
        return perimeter, jnp.abs(jnp.dot(total, normal) / 2.0)

    def ellipse_a(perimeter, area, a_guess):
        def resid(a):
            b = area / (jnp.pi * a)
            return perimeter - 4.0 * a * ellipe(1.0 - (b / a) ** 2)
        dr = jax.grad(resid)

        def body(_, a):
            d = dr(a)
            step = jnp.where(jnp.abs(d) > 1e-300, resid(a) / d, 0.0)
            return a - step
        return jax.lax.fori_loop(0, _ELLIPSE_ITERS, body, a_guess)

    def per_slice(a_guess, inputs):
        phi_i, tangent, axis_xyz = inputs
        pts = cross_section(phi_i, tangent, axis_xyz)
        perimeter, area = polygon_perimeter_area(pts)
        a_sol = ellipse_a(perimeter, area, a_guess)
        b_sol = area / (jnp.pi * a_sol)
        elong = jnp.maximum(a_sol, b_sol) / jnp.minimum(a_sol, b_sol)
        return a_sol, elong

    _, elongations = jax.lax.scan(per_slice, 1.0,
                                  (phi1d, tangents, axes_xyz))
    return jnp.max(elongations), elongations
