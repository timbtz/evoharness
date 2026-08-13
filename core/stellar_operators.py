"""Deterministic, symmetry-preserving structural boundary transformations."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


def _arrays(boundary: dict) -> tuple[np.ndarray, np.ndarray]:
    rc = np.asarray(boundary["r_cos"], dtype=float)
    zs = np.asarray(boundary["z_sin"], dtype=float)
    if rc.ndim != 2 or rc.shape != zs.shape or rc.shape[1] % 2 != 1:
        raise ValueError("invalid Fourier matrices")
    if not np.isfinite(rc).all() or not np.isfinite(zs).all():
        raise ValueError("non-finite Fourier coefficient")
    return rc, zs


def _boundary_like(boundary: dict, rc: np.ndarray, zs: np.ndarray) -> dict:
    out = copy.deepcopy(boundary)
    out["r_cos"], out["z_sin"] = rc.tolist(), zs.tolist()
    return out


def geometry_screen(boundary: dict, max_abs: float = 5.0) -> tuple[bool, str]:
    """Exact cheap invariants; equilibrium-dependent checks stay in physics."""
    try:
        rc, zs = _arrays(boundary)
        nfp = int(boundary["n_field_periods"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, str(exc)
    if nfp < 1 or nfp > 20:
        return False, "n_field_periods outside [1, 20]"
    ntor = (rc.shape[1] - 1) // 2
    if rc[0, ntor] <= .1:
        return False, "non-positive major radius"
    if max(float(np.abs(rc).max()), float(np.abs(zs).max())) > max_abs:
        return False, "coefficient magnitude cap exceeded"
    # Stellarator-symmetry pins used by the task contract.
    if np.any(np.abs(rc[0, :ntor]) > 1e-12) or np.any(np.abs(zs[0, :ntor + 1]) > 1e-12):
        return False, "forbidden m=0 symmetry coefficient"
    return True, "ok"


def truncate(boundary: dict, max_m: int, max_n: int) -> dict:
    """Reveal a boundary's low-order structural core without changing its grid."""
    if max_m < 0 or max_n < 0:
        raise ValueError("mode limits must be nonnegative")
    rc, zs = _arrays(boundary)
    ntor = (rc.shape[1] - 1) // 2
    mask = np.zeros_like(rc, dtype=bool)
    mask[:min(max_m + 1, rc.shape[0]),
         max(0, ntor - max_n):min(rc.shape[1], ntor + max_n + 1)] = True
    return _boundary_like(boundary, rc * mask, zs * mask)


def expand_grid(boundary: dict, max_m: int, max_n: int) -> dict:
    """Zero-pad around n=0, preserving every existing coefficient exactly."""
    rc, zs = _arrays(boundary)
    old_n = (rc.shape[1] - 1) // 2
    if max_m < rc.shape[0] - 1 or max_n < old_n:
        raise ValueError("expand_grid cannot remove modes")
    shape = (max_m + 1, 2 * max_n + 1)
    rr, zz = np.zeros(shape), np.zeros(shape)
    offset = max_n - old_n
    rr[:rc.shape[0], offset:offset + rc.shape[1]] = rc
    zz[:zs.shape[0], offset:offset + zs.shape[1]] = zs
    return _boundary_like(boundary, rr, zz)


@dataclass(frozen=True)
class ModeBand:
    max_m: int
    max_n: int
    amplitude: float
    phase_seed: int


def activate_band(boundary: dict, band: ModeBand) -> dict:
    """Activate only the newly exposed support shell with coordinated modes."""
    if band.amplitude <= 0 or band.amplitude > .05:
        raise ValueError("structural band amplitude must be in (0, .05]")
    rc0, _ = _arrays(boundary)
    old_m, old_n = rc0.shape[0] - 1, (rc0.shape[1] - 1) // 2
    out = expand_grid(boundary, max(old_m, band.max_m), max(old_n, band.max_n))
    rc, zs = _arrays(out)
    rng = np.random.default_rng(band.phase_seed)
    scale = float(rc[0, rc.shape[1] // 2])
    center = rc.shape[1] // 2
    for m in range(rc.shape[0]):
        for j in range(rc.shape[1]):
            n = j - center
            if m <= old_m and abs(n) <= old_n:
                continue
            if m == 0 and n <= 0:
                continue
            # Coordinated R/Z quadrature, decayed by spectral order.
            amp = band.amplitude * scale / max(1, m + abs(n))
            sign = -1.0 if rng.integers(2) else 1.0
            rc[m, j] = sign * amp
            if not (m == 0 and n <= 0):
                zs[m, j] = -sign * amp
    return _boundary_like(out, rc, zs)
