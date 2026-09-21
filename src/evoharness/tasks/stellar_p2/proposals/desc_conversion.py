"""Boundary dict <-> DESC FourierRZToroidalSurface converter.

Transcribed from the pinned image's
`constellaration/geometry/surface_utils_desc.py` (constellaration==0.2.6,
read via `docker run ... cat`, 2026-08-16) so the DESC-only venv
(.venv-desc, no constellaration installed) can convert boundaries. The
math is identical: DESC's double-Fourier product basis <-> the single-angle
cos(m*theta - n*NFP*phi) basis via desc.vmec_utils.ptolemy_identity_fwd/rev.

Convention (constellaration SurfaceRZFourier JSON):
  r_cos/z_sin are (mpol+1) x (2*ntor+1) matrices; column j holds toroidal
  mode n = j - ntor; stellarator-symmetric surfaces have r_sin=z_cos=None
  and r_cos[0, n<0] == z_sin[0, n<0] == 0 (enforced on the way back).
"""
from __future__ import annotations

import numpy as np
from desc import geometry as desc_geometry
from desc import vmec_utils as desc_vmec_utils


def to_desc_surface(boundary: dict) -> desc_geometry.FourierRZToroidalSurface:
    rc = np.asarray(boundary["r_cos"], float)
    zs = np.asarray(boundary["z_sin"], float)
    if not boundary.get("is_stellarator_symmetric", True):
        raise NotImplementedError("only stellarator-symmetric boundaries")
    nfp = int(boundary["n_field_periods"])
    mpol1, ncols = rc.shape
    ntor = (ncols - 1) // 2

    mm, nn = np.meshgrid(np.arange(mpol1), np.arange(-ntor, ntor + 1),
                         indexing="ij")
    poloidal = mm.ravel().astype(int)
    toroidal = nn.ravel().astype(int)
    rmnc = rc.ravel()
    zmns = zs.ravel()
    rmns = np.zeros_like(rmnc)
    zmnc = np.zeros_like(zmns)

    # drop (m=0, n<0): not independent modes under stellarator symmetry
    keep = ~((poloidal == 0) & (toroidal < 0))
    poloidal, toroidal = poloidal[keep], toroidal[keep]
    rmnc, zmns = rmnc[keep], zmns[keep]
    rmns, zmnc = rmns[keep], zmnc[keep]

    m, n, r_lmn = desc_vmec_utils.ptolemy_identity_fwd(
        poloidal, toroidal, s=rmns, c=rmnc)
    m2, n2, z_lmn = desc_vmec_utils.ptolemy_identity_fwd(
        poloidal, toroidal, s=zmns, c=zmnc)
    assert np.array_equal(m, m2) and np.array_equal(n, n2)

    modes = np.vstack((m, n)).T.astype(int)
    return desc_geometry.FourierRZToroidalSurface(
        R_lmn=np.asarray(r_lmn).ravel(),
        Z_lmn=np.asarray(z_lmn).ravel(),
        modes_R=modes,
        modes_Z=modes,
        NFP=nfp,
        sym=True,
        check_orientation=False,
    )


def from_desc_surface(surface: desc_geometry.FourierRZToroidalSurface) -> dict:
    nfp = int(surface.NFP)
    if not surface.sym:
        raise NotImplementedError("only stellarator-symmetric surfaces")

    m_r = surface.R_basis.modes[:, 1]
    n_r = surface.R_basis.modes[:, 2]
    m_out_r, n_out_r, _sin_r, cos_r = desc_vmec_utils.ptolemy_identity_rev(
        m_1=m_r, n_1=n_r, x=np.expand_dims(surface.R_lmn, axis=0))
    m_z = surface.Z_basis.modes[:, 1]
    n_z = surface.Z_basis.modes[:, 2]
    m_out_z, n_out_z, sin_z, _cos_z = desc_vmec_utils.ptolemy_identity_rev(
        m_1=m_z, n_1=n_z, x=np.expand_dims(surface.Z_lmn, axis=0))

    mpol = int(max(m_out_r.max(initial=0), m_out_z.max(initial=0)))
    ntor = int(max(np.abs(n_out_r).max(initial=0),
                   np.abs(n_out_z).max(initial=0)))
    rc = np.zeros((mpol + 1, 2 * ntor + 1))
    zs = np.zeros((mpol + 1, 2 * ntor + 1))
    for m, n, v in zip(m_out_r, n_out_r, cos_r[0, :]):
        rc[int(m), int(n) + ntor] = v
    for m, n, v in zip(m_out_z, n_out_z, sin_z[0, :]):
        zs[int(m), int(n) + ntor] = v

    # enforce the symmetric-surface convention the evaluator validates
    rc[0, :ntor] = 0.0
    zs[0, :ntor] = 0.0

    return {"r_cos": rc.tolist(), "z_sin": zs.tolist(),
            "r_sin": None, "z_cos": None,
            "n_field_periods": nfp, "is_stellarator_symmetric": True}
