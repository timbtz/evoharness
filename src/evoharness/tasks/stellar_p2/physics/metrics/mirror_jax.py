"""Edge magnetic mirror ratio (P2 constraint index 3) in JAX.

Reproduces the pinned chain exactly (constellaration 0.2.6):
  mhd/magnetics_utils.magnetic_mirror_ratio
    -> _magnetic_field_strength_nyquist_resolution
         theta = linspace(0, 2pi, 2*mpol+6, endpoint=True)
         phi   = linspace(0, 2pi/nfp/(1+int(not lasym)), 2*ntor+4, endpoint=True)
         B(s,theta,phi) from bmnc.T[1:, :] (half mesh, first row dropped)
         radially interp1d(linear, fill_value="extrapolate") on
         normalized_toroidal_flux_half_grid_mesh[1:]
    -> per-surface (Bmax - Bmin)/(Bmax + Bmin)
  forward_model: edge value = InterpolatedUnivariateSpline of the profile at
  rho = 1.0; the profile's rho grid ends at exactly 1.0, and an interpolating
  spline passes through its nodes, so the edge value equals the s = 1.0 row —
  verified numerically in the parity test, not assumed.

Only the s = 1.0 surface is therefore needed: its Fourier coefficients are the
LINEAR EXTRAPOLATION of the last two half-mesh rows of bmnc.

Non-smoothness note (same class as lgradb's min): Bmax/Bmin are exact
max/min over the fixed Nyquist grid; jax.grad returns the subgradient at the
achieved arg-extremum. The argext can switch under finite boundary steps —
the established estimator (analytic d(mirror)/d(bmnc) contracted against FD
of the bmnc arrays, which are smooth in the boundary) is the right one, per
the 2026-08-05 methodology. No softmin is applied (exact by default, like
lgradb_jax with smoothing=0).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def edge_mirror_ratio(bmnc, xm_nyq, xn_nyq, s_half, mpol, ntor, nfp, lasym):
    """Edge magnetic mirror ratio from wout arrays.

    Args:
      bmnc: (n_modes_nyq, ns) as stored in the wout (pre-transpose).
      xm_nyq, xn_nyq: Nyquist mode numbers (xn includes the nfp factor).
      s_half: normalized_toroidal_flux_half_grid_mesh, full length ns
              (element 0 unused, as in the pinned code).
      mpol, ntor, nfp: VMEC integers. lasym: bool.
    """
    b = bmnc.T[1:, :]                     # (ns-1, n_modes) half mesh
    x = s_half[1:]                        # (ns-1,)
    slope = (b[-1] - b[-2]) / (x[-1] - x[-2])
    b_edge = b[-1] + slope * (1.0 - x[-1])   # linear extrapolation to s=1.0

    n_theta = 2 * int(mpol) + 6
    n_phi = 2 * int(ntor) + 4
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=True)
    phi_ub = 2.0 * jnp.pi / nfp / (1 + int(not lasym))
    phi = jnp.linspace(0.0, phi_ub, n_phi, endpoint=True)

    ang = (xm_nyq[None, None, :] * theta[:, None, None]
           - xn_nyq[None, None, :] * phi[None, :, None])
    B = jnp.sum(b_edge[None, None, :] * jnp.cos(ang), axis=-1)
    b_max = jnp.max(B)
    b_min = jnp.min(B)
    return (b_max - b_min) / (b_max + b_min)
