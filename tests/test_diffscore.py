"""Golden tests for experiments/diffscore (Plan 2).

These need the pinned eval stack (jax + constellaration) and the oracle cache,
so they run inside the eval image and SKIP on the host venv:

  docker run --rm --cpus=2 --memory=2500m \
      -v <repo>:/work -w /work evoharness-stellar-eval \
      sh -c "pip install -q pytest && python -m pytest tests/test_diffscore.py -q"

The oracle cache is built by:
  python -m experiments.diffscore.difftest --build-cache --n 200   (in-image)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("constellaration")

from experiments.diffscore import (  # noqa: E402
    elongation_jax, feasibility, margins_jax, qi_jax)
from experiments.diffscore.difftest import load_cache  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / "diffscore" / "oracle_cache" / "index.jsonl"

needs_cache = pytest.mark.skipif(not CACHE.exists(),
                                 reason="oracle cache not built")


def _rows(n=None, strata=()):
    rows = load_cache(strata)
    return rows[:n] if n else rows


# -- margins ----------------------------------------------------------------

@needs_cache
def test_margins_value_parity():
    """Aspect (boundary-only) parity <= 1e-12; wout-input metrics exact to
    1e-10; assembled violations track the oracle to 1e-9."""
    rows = _rows()
    assert len(rows) >= 100, "cache too small for a meaningful parity claim"
    worst = {"aspect": 0.0, "iota": 0.0, "mirror": 0.0, "viol": 0.0}
    for r in rows:
        z = np.load(r["npz"])
        nfp = int(z["nfp"])
        a = float(margins_jax.aspect_ratio(
            np.array(r["boundary"]["r_cos"]), np.array(r["boundary"]["z_sin"]),
            r["boundary"]["n_field_periods"]))
        ei = float(margins_jax.edge_iota_over_nfp(z["iotaf"], nfp))
        mi = float(margins_jax.edge_mirror_ratio(
            z["bmnc"], z["normalized_toroidal_flux_half_grid_mesh"],
            z["normalized_toroidal_flux_full_grid_mesh"], z["xm_nyq"],
            z["xn_nyq"], int(z["mpol"]), int(z["ntor"]), nfp))
        worst["aspect"] = max(worst["aspect"], abs(a - r["aspect_ratio"]))
        worst["iota"] = max(worst["iota"], abs(
            ei - r["edge_rotational_transform_over_n_field_periods"]))
        worst["mirror"] = max(worst["mirror"],
                              abs(mi - r["edge_magnetic_mirror_ratio"]))
        v = np.asarray(margins_jax.normalized_violations(
            a, ei, r["qi"], mi, r["max_elongation"]))
        worst["viol"] = max(worst["viol"],
                            float(np.max(np.abs(v - np.array(r["violations"])))))
    assert worst["aspect"] <= 1e-12, worst
    assert worst["iota"] <= 1e-10, worst
    assert worst["mirror"] <= 1e-10, worst
    assert worst["viol"] <= 1e-9, worst


@needs_cache
def test_margins_elongation_parity():
    """Newton-based elongation vs the fsolve-based oracle on a stratified
    subset (slowest margin; full sweep lives in difftest --margins)."""
    rows = _rows(30)
    worst = 0.0
    for r in rows:
        z = np.load(r["npz"])
        el, _ = margins_jax.max_elongation_wout(
            z["rmnc"], z["zmns"], z["xm"], z["xn"], z["raxis_cc"],
            z["zaxis_cs"], int(z["mpol"]), int(z["ntor"]), int(z["nfp"]))
        worst = max(worst, abs(float(el) - r["max_elongation"]))
    assert worst <= 1e-8, worst


@needs_cache
def test_margins_grad_vs_fd():
    rows = _rows(6)
    for r in rows:
        bd = r["boundary"]
        rc, zs = np.array(bd["r_cos"]), np.array(bd["z_sin"])
        g_rc, _ = margins_jax.aspect_grad(rc, zs, bd["n_field_periods"])
        err = margins_jax._fd_check(
            lambda x: margins_jax.aspect_ratio(x, zs, bd["n_field_periods"]),
            rc, g_rc)
        assert err <= 1e-6, (r["key"], err)
    # elongation gradient on one boundary (expensive)
    z = np.load(rows[0]["npz"])
    rl = np.asarray(z["rmnc"]).T[-1, :].copy()

    def elong_of_rl(x):
        v, _ = elongation_jax.max_elongation(
            x, np.asarray(z["zmns"]).T[-1, :], z["xm"], z["xn"],
            z["raxis_cc"], z["zaxis_cs"], 2 * int(z["mpol"]) + 6,
            2 * int(z["ntor"]) + 4, int(z["nfp"]))
        return v
    g = jax.grad(elong_of_rl)(rl)
    err = margins_jax._fd_check(elong_of_rl, rl, g)
    assert err <= 1e-5, err


def test_asymmetric_boundary_rejected():
    with pytest.raises(NotImplementedError):
        margins_jax.aspect_from_boundary(
            {"r_cos": [[0, 1, 0.1]], "z_sin": [[0, 0, 0.1]],
             "r_sin": [[0, 0, 0.01]], "z_cos": None, "n_field_periods": 3,
             "is_stellarator_symmetric": False})


# -- qi ---------------------------------------------------------------------

def _qi_args(z):
    return (z["bmnc_b"], z["xm_b"], z["xn_b"], z["boozer_iota"], z["s_in"],
            np.atleast_1d(z["s_b"]), int(z["nfp"]), float(z["bmnc_11"]))


@needs_cache
def test_qi_stagewise_parity():
    """squash/stretch, weights and bounce points vs the pinned qi.py internals
    at smoothing=0 on 20 boundaries."""
    from constellaration.omnigeneity import qi as qi_ref
    from scipy import interpolate
    rows = [r for r in _rows() if r["qi"] is not None][:20]
    assert len(rows) == 20
    worst_sq, worst_w, worst_bp = 0.0, 0.0, 0.0
    for r in rows:
        z = np.load(r["npz"])
        nfp = int(z["nfp"])
        iota_b = float(interpolate.UnivariateSpline(
            z["s_in"], z["boozer_iota"], k=1, s=0)(np.atleast_1d(z["s_b"]))[0])
        phi_start = np.pi / nfp if float(z["bmnc_11"]) < 0 else 0.0
        nphi, nalpha = 601, 75
        phis = np.linspace(phi_start, phi_start + 2 * np.pi / nfp, nphi)
        phis2d = np.tile(phis, (nalpha, 1)).T
        thetas2d = (np.tile(np.linspace(-iota_b * phi_start,
                                        -iota_b * phi_start + 2 * np.pi,
                                        nalpha), (nphi, 1)) + iota_b * phis2d)
        # chunked mode sum: the full (n_modes, nphi, nalpha) tensor is ~GBs
        modb = np.zeros((nphi, nalpha))
        for k0 in range(0, z["xm_b"].size, 128):
            sl = slice(k0, k0 + 128)
            angle = (z["xm_b"][sl, None, None] * thetas2d[None, ...]
                     - z["xn_b"][sl, None, None] * phis2d[None, ...])
            modb += np.sum(np.cos(angle) * z["bmnc_b"][sl, 0][:, None, None],
                           axis=0)
        modb = (modb - modb.min()) / (modb.max() - modb.min())
        for ialpha in range(0, nalpha, 15):
            line = modb[:, ialpha]
            mi = int(np.argmin(line))
            if mi in (0, len(line) - 1):
                continue
            left = qi_ref._stretch_left_side(
                qi_ref._squash_left_side(line[:mi + 1]))
            right = qi_ref._stretch_right_side(
                qi_ref._squash_right_side(line[mi:]))
            ref_sq = np.concatenate([left[:-1], right])
            jax_sq = np.asarray(qi_jax.squash_stretch_line(line))
            worst_sq = max(worst_sq, float(np.max(np.abs(jax_sq - ref_sq))))
            w_ref = (phis[-1] - phis[0]) / float(interpolate.UnivariateSpline(
                phis, (line - ref_sq) ** 2).integral(phis[0], phis[-1]))
            w_jax = float(qi_jax.fieldline_weight(phis, (line - jax_sq) ** 2))
            worst_w = max(worst_w, abs(w_jax - w_ref) / abs(w_ref))
            Bjs = np.linspace(0, 1, 401)
            bp_jax = np.asarray(qi_jax.bounce_points_line(phis, jax_sq, Bjs))
            for j in range(0, 401, 40):
                ref_l, ref_r = qi_ref._find_bounce_points(
                    phi=phis, modb=ref_sq, modb_star=Bjs[j],
                    modb_max_on_flux_surface=1.0,
                    modb_min_on_flux_surface=0.0)
                worst_bp = max(worst_bp,
                               abs(bp_jax[j, 0] - ref_l),
                               abs(bp_jax[j, 1] - ref_r))
    assert worst_sq <= 1e-12, worst_sq
    assert worst_w <= 1e-9, worst_w
    assert worst_bp <= 1e-9, worst_bp


@needs_cache
def test_qi_end_to_end_parity():
    """|delta log10 qi| <= 0.05 required; exact mode achieves ~1e-6. The
    cached r['qi'] IS the pinned reference (forward_model runs the same qi.py
    on the same boozer arrays; the duck-typed stub reproduced the saved
    residuals bit-exactly during development), so no in-test reference re-run
    — that re-run materializes multi-GB mode tensors and OOMs the box."""
    rows = [r for r in _rows() if r["qi"] is not None][:25]
    worst = 0.0
    for r in rows:
        z = np.load(r["npz"])
        q = float(np.sum(np.asarray(qi_jax.qi_residual(*_qi_args(z))) ** 2))
        worst = max(worst, abs(np.log10(q) - np.log10(r["qi"])))
    assert worst <= 1e-4, worst


@needs_cache
def test_qi_grad_finite_and_fd():
    rows = [r for r in _rows() if r["qi"] is not None][:3]
    for r in rows:
        z = np.load(r["npz"])
        args = _qi_args(z)
        g = jax.grad(lambda b: qi_jax.qi_metric(b, *args[1:]))(
            jax.numpy.asarray(args[0]))
        assert np.all(np.isfinite(np.asarray(g))), r["key"]
        rng = np.random.default_rng(0)
        d = rng.standard_normal(args[0].shape)
        d /= np.linalg.norm(d)
        gd = float(jax.numpy.sum(g * d))
        h = 1e-6 * float(np.max(np.abs(args[0])))
        fd = (float(qi_jax.qi_metric(args[0] + h * d, *args[1:]))
              - float(qi_jax.qi_metric(args[0] - h * d, *args[1:]))) / (2 * h)
        # the reference pipeline is a.e.-differentiable (argmin/cummin/crossing
        # selection); FD averages across the kinks a step of size h crosses,
        # autodiff returns the one-sided slope — measured agreement is ~1e-5
        # on kink-free directions, up to ~3e-3 across bounce-birth kinks
        assert abs(fd - gd) / max(abs(fd), abs(gd)) <= 1e-2, (r["key"], fd, gd)


@needs_cache
def test_lgradb_score_parity_and_grad():
    """The P2 objective (min normalized L_gradB) from wout arrays: value
    parity vs the pinned metric and a finite gradient w.r.t. bmnc."""
    from experiments.diffscore import lgradb_jax
    rows = _rows(10)
    for r in rows:
        z = np.load(r["npz"])
        args = dict(
            rmnc=z["rmnc"], zmns=z["zmns"], gmnc=z["gmnc"], bmnc=z["bmnc"],
            bsupumnc=z["bsupumnc"], bsupvmnc=z["bsupvmnc"], xm=z["xm"],
            xn=z["xn"], xm_nyq=z["xm_nyq"], xn_nyq=z["xn_nyq"],
            ns=int(z["ns"]), aminor_p=float(z["Aminor_p"]),
            mpol=int(z["mpol"]), ntor=int(z["ntor"]), nfp=int(z["nfp"]),
            lasym=bool(z["lasym"]))
        val = float(lgradb_jax.min_normalized_l_grad_b(**args))
        ref = r["minimum_normalized_magnetic_gradient_scale_length"]
        assert abs(val - ref) <= 1e-9 * max(1.0, abs(ref)), (r["key"], val,
                                                            ref)
    # gradient check on the last row's arrays (args still in scope)
    def f(b):
        a = dict(args)
        a["bmnc"] = b
        return lgradb_jax.min_normalized_l_grad_b(**a)
    g = jax.grad(f)(jax.numpy.asarray(args["bmnc"]))
    assert np.all(np.isfinite(np.asarray(g)))


# -- feasibility / dominance ------------------------------------------------

def test_soft_feasibility_converges_to_max():
    v = np.array([-0.5, 0.003, -0.02, 0.0091, -0.1])
    exact = max(v.max(), 0.0)
    for tau in (1e-3, 1e-4, 1e-5):
        s = float(feasibility.soft_feasibility(v, tau))
        assert s >= exact  # logsumexp is an upper bound
    assert abs(float(feasibility.soft_feasibility(v, 1e-5)) - exact) < 1e-4


def test_strict_dominance_guard():
    """No smooth-only fitness may ever outrank an officially feasible score."""
    official_worst = feasibility.rank_key(True, 1e-9, 0.0)
    for smooth in (0.0, 0.5, 0.63, 10.0, float("inf")):
        assert official_worst > feasibility.rank_key(False, 0.0, smooth)
    # and among officially feasible, the official score decides
    assert (feasibility.rank_key(True, 0.64, 0.0)
            > feasibility.rank_key(True, 0.63, 99.0))


@needs_cache
def test_far_infeasible_rows_parity():
    """The 'far' stratum (violations >> tol) must not break any pipeline."""
    rows = _rows(strata=("far",))
    for r in rows[:3]:
        z = np.load(r["npz"])
        a = float(margins_jax.aspect_ratio(
            np.array(r["boundary"]["r_cos"]), np.array(r["boundary"]["z_sin"]),
            r["boundary"]["n_field_periods"]))
        assert abs(a - r["aspect_ratio"]) <= 1e-10
        v = np.asarray(margins_jax.normalized_violations(
            a, r["edge_rotational_transform_over_n_field_periods"], r["qi"],
            r["edge_magnetic_mirror_ratio"], r["max_elongation"]))
        f = float(margins_jax.feasibility(v))
        assert abs(f - r["feasibility"]) <= 1e-9


# -- the in-container port (task.py _GRAD_SRC) ------------------------------

@needs_cache
def test_template_aspect_matches_jax_reference():
    """The numpy aspect + analytic gradient shipped INSIDE the train template
    must reproduce this package's autodiff reference — the template cannot
    import jax (it runs in the process that forks the eval pool)."""
    from tasks.stellar_p2 import task as st
    worst_v, worst_g = 0.0, 0.0
    for r in _rows(6):
        bd = r["boundary"]
        rc, zs = np.array(bd["r_cos"]), np.array(bd["z_sin"])
        nfp = bd["n_field_periods"]
        a, g_rc, g_zs = st._aspect_full(rc, zs, nfp)
        worst_v = max(worst_v, abs(a - float(margins_jax.aspect_ratio(rc, zs, nfp))))
        j_rc, j_zs = margins_jax.aspect_grad(rc, zs, nfp)
        scale = max(float(np.abs(np.asarray(j_rc)).max()), 1e-30)
        worst_g = max(worst_g,
                      float(np.abs(g_rc - np.asarray(j_rc)).max()) / scale,
                      float(np.abs(g_zs - np.asarray(j_zs)).max()) / scale)
    assert worst_v <= 1e-12, worst_v
    assert worst_g <= 1e-8, worst_g
