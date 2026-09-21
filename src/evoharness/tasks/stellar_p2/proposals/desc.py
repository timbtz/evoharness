"""DESC proposing lane (wiki/06-desc-toolkit.md): O(1)-gradient boundary steps.

DESC PROPOSES, THE PINNED EVALUATOR ADJUDICATES. Nothing here scores anything;
the output boundary JSON must be handed to tasks/stellar_p2/task.verify_boundary.

Formulation (per the 2026-08-16 assessment):
  objective terms (lsq-auglag weighted least squares, Proxima's own split):
    - BScaleLength on an LCFS grid with bounds=(floor, inf), normalize off:
      a LOWER BOUND ON EVERY GRID POINT of L_grad(B) — raises the min without
      differentiating through the switching argmin. floor = (1+delta)*current.
    - AspectRatio bounds (0, 10); Elongation bounds (0, 5);
      MirrorRatio bounds (0, 0.2)         [P2 bounds from problems.py]
    - RotationalTransform bounds keeping |iota| >= 0.25*nfp (DESC sign kept
      from the solved equilibrium; constellaration's is flipped).
    - optional Omnigenity(eq, field, field_fixed=False) with a generic
      OmnigenousField co-optimized alongside the equilibrium (--omnigenity).
  constraints: FixBoundaryR(M=0,N=0), CurrentDensity (vacuum), FixPressure,
      FixCurrent, FixPsi  [Proxima's create_desc_constraints verbatim]

Run in the DESC-only venv (persistent process; JIT is paid once):
  OMP_NUM_THREADS=2 nice -n 10 .venv-desc/bin/python -m \
      evoharness.tasks.stellar_p2.proposals.desc \
      --boundary-file src/evoharness/tasks/stellar_p2/seed_bank.json --seed-index 0 \
      --M 3 --N 3 --maxiter 2 --out .local/runs/desc-probe-1/proposal.json
  --roundtrip-only skips all solves (converter validation).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

def to_desc_surface(boundary: dict):
    from evoharness.tasks.stellar_p2.proposals.desc_conversion import to_desc_surface as convert
    return convert(boundary)


def from_desc_surface(surface):
    from evoharness.tasks.stellar_p2.proposals.desc_conversion import from_desc_surface as convert
    return convert(surface)


def load_boundary(path: str, seed_index: int | None) -> dict:
    payload = json.loads(Path(path).read_text())
    if "seeds" in payload:
        return payload["seeds"][seed_index or 0]["boundary"]
    return payload.get("boundary", payload)


def fresh_nae_boundary(nfp: int, aspect: float, elongation: float,
                       axis_r: float, axis_z: float) -> dict:
    """Fresh-start rotating-ellipse boundary (L7 vehicle, no bank seed).

    Conventions transcribed from constellaration's SurfaceRZFourier (single-
    angle cos(m*theta - n*NFP*phi) basis; column j <-> n = j - ntor) — NOT
    imported. Shape: circular base of minor radius a = 1/aspect, plus an
    m=1, n=1 helical term delta = a*(k-1)/(k+1) that rotates the ellipse with
    the field period (elongation k), plus axis modulation r_axis*cos(NFP phi)
    and torsion z_axis*sin(NFP phi). wiki/05's embeddability wall: the axis
    torsion knob must stay <= ~0.25 or the NAE->VMEC transfer cliffs.
    """
    axis_z = min(axis_z, 0.25)
    amin = 1.0 / aspect
    delta = amin * (elongation - 1.0) / (elongation + 1.0)
    ntor = 2
    rc = np.zeros((3, 2 * ntor + 1))
    zs = np.zeros((3, 2 * ntor + 1))
    n0 = ntor
    rc[0, n0] = 1.0
    rc[0, n0 + 1] = axis_r
    zs[0, n0 + 1] = axis_z
    rc[1, n0] = amin
    zs[1, n0] = amin
    rc[1, n0 + 1] = delta
    zs[1, n0 + 1] = -delta
    return {"r_cos": rc.tolist(), "z_sin": zs.tolist(),
            "r_sin": None, "z_cos": None,
            "n_field_periods": int(nfp), "is_stellarator_symmetric": True}


def fit_omnigenous_field(eq, lcfs_grid, fit_iters: int, log=print):
    """Fit an OmnigenousField to the SEED equilibrium so the Omnigenity term
    starts near zero and only penalizes DEGRADATION during the boundary
    optimization. Two stages: (1) B_lm knots from the equilibrium's own |B|
    range on the LCFS (monotonic by construction); (2) a small field-only
    least-squares fit (eq_fixed=True) of the x_lmn deformation parameters."""
    from desc.magnetic_fields import OmnigenousField
    from desc.objectives import ObjectiveFunction, Omnigenity
    from desc.optimize import Optimizer

    nfp = int(eq.NFP)
    modB = np.asarray(eq.compute(["|B|"], grid=lcfs_grid)["|B|"])
    knots = np.array([float(np.min(modB)), float(np.median(modB)),
                      float(np.max(modB))])
    # strict monotonicity guard (median can collide with an extremum)
    eps = 1e-6 * max(1.0, knots[2])
    knots[1] = min(max(knots[1], knots[0] + eps), knots[2] - eps)
    log(f"fit-well: |B| knots {knots.round(6).tolist()} "
        f"(mirror {(knots[2]-knots[0])/(knots[2]+knots[0]):.4f})")
    field = OmnigenousField(L_B=0, M_B=3, L_x=1, M_x=1, N_x=1,
                            NFP=nfp, helicity=(0, nfp), B_lm=knots)
    if fit_iters > 0:
        obj = ObjectiveFunction((Omnigenity(
            eq=eq, field=field, eq_fixed=True, field_fixed=False,
            eq_grid=lcfs_grid, weight=1.0),))
        (fitted,), _ = Optimizer("lsq-exact").optimize(
            things=[field], objective=obj, maxiter=fit_iters, verbose=1,
            copy=True)
        # BLOW-UP GUARD (2026-08-17): the qi-repair probe died because this
        # fit "converged" along a flat direction to |x_lmn| ~ 3e14, and the
        # main objective build around the broken field then OOM-killed the
        # process. x_lmn are Fourier coefficients of an angle deformation —
        # anything beyond ~pi is unphysical wrapping.
        x = np.asarray(fitted.x_lmn)
        B = np.asarray(fitted.B_lm)
        if (np.all(np.isfinite(x)) and np.all(np.isfinite(B))
                and float(np.abs(x).max(initial=0.0)) <= np.pi):
            field = fitted
            log("fit-well: x_lmn fitted "
                f"(|x|_max {float(np.abs(x).max(initial=0.0)):.4f})")
        else:
            field.x_lmn = np.zeros_like(np.asarray(field.x_lmn))
            log(f"fit-well: REJECTED diverged x_lmn fit (|x|_max "
                f"{float(np.abs(x).max(initial=0.0)):.3e}) — keeping "
                "knots-only field with x_lmn = 0")
    return field


def solve_force_chunked(eq, chunk: int, verbose: int = 2):
    """Force solve with a memory-bounded jacobian (jac memory ~ m0 + m1*chunk).
    The 2026-08-17 synth run was OOM-killed in the first full-width force
    jacobian at M=N=6 on a ~2 GB-free box; chunking trades wall-time for RAM."""
    from desc.objectives import get_equilibrium_objective
    obj = get_equilibrium_objective(eq=eq, mode="force", jac_chunk_size=chunk)
    eq, _ = eq.solve(objective=obj, verbose=verbose)
    return eq


def build_and_solve(surf, M: int, N: int, psi: float, chunk: int,
                    log=print):
    """Equilibrium via resolution continuation: cold-solve at M=N<=4, then
    step up to the target and re-solve warm. The peak-memory jacobian then
    runs on a warm state (fewer iterations at full width)."""
    from desc.equilibrium import Equilibrium
    m0 = min(4, M)
    n0 = min(4, N)
    eq = Equilibrium(Psi=psi, surface=surf, M=m0, N=n0,
                     check_orientation=False)
    log(f"force solve at M={m0} N={n0} (chunk {chunk})...")
    eq = solve_force_chunked(eq, chunk)
    if (M, N) != (m0, n0):
        log(f"continuation: raising resolution to M={M} N={N}...")
        eq.change_resolution(M=M, N=N)
        eq = solve_force_chunked(eq, chunk)
    return eq


def field_from_donor(donor_boundary: dict, platform_eq, platform_grid,
                     M: int, N: int, psi: float, fit_iters: int,
                     jac_chunk: int = 16, log=print):
    """Fit an OmnigenousField to a DONOR boundary (a true QI design, so the
    fit is tight), then re-target it onto the platform: keep the |B|(eta)
    well SHAPE, rescale to the platform's field strength, cap the mirror
    depth at the P2 bound, rebuild at the platform's NFP/helicity, and seed
    x_lmn from the donor when sane (the main optimization keeps the field
    free, so x_lmn only needs to start in the right basin)."""
    from desc.equilibrium import Equilibrium
    from desc.grid import LinearGrid
    from desc.magnetic_fields import OmnigenousField

    donor_surf = to_desc_surface(donor_boundary)
    log(f"well-from: solving donor equilibrium (nfp={donor_surf.NFP}, "
        f"M=N={M})...")
    donor_eq = build_and_solve(donor_surf, M, N, psi, chunk=jac_chunk,
                               log=log)
    donor_grid = LinearGrid(rho=np.array([1.0]), M=4 * donor_eq.M,
                            N=4 * donor_eq.N, NFP=int(donor_eq.NFP),
                            sym=False)
    donor_field = fit_omnigenous_field(donor_eq, donor_grid, fit_iters,
                                       log=log)

    knots = np.asarray(donor_field.B_lm, float)[:3].copy()
    modB_p = np.asarray(platform_eq.compute(["|B|"], grid=platform_grid)
                        ["|B|"])
    scale = float(np.mean(modB_p)) / float(np.mean(knots))
    knots *= scale
    mid = 0.5 * (knots[0] + knots[2])
    mirror = (knots[2] - knots[0]) / (knots[2] + knots[0])
    if mirror > 0.19:
        knots = mid + (knots - mid) * (0.19 / mirror)
        log(f"well-from: donor mirror {mirror:.4f} capped to 0.19")
    nfp_p = int(platform_eq.NFP)
    x_donor = np.asarray(donor_field.x_lmn)
    field = OmnigenousField(L_B=0, M_B=3, L_x=1, M_x=1, N_x=1,
                            NFP=nfp_p, helicity=(0, nfp_p), B_lm=knots)
    if (np.all(np.isfinite(x_donor))
            and float(np.abs(x_donor).max(initial=0.0)) <= np.pi
            and x_donor.shape == np.asarray(field.x_lmn).shape):
        field.x_lmn = x_donor.copy()
        log(f"well-from: transferred donor x_lmn (|x|_max "
            f"{float(np.abs(x_donor).max(initial=0.0)):.4f})")
    else:
        log("well-from: donor x_lmn not transferable — starting at 0")
    log(f"well-from: platform field knots {knots.round(6).tolist()} "
        f"(mirror {(knots[2]-knots[0])/(knots[2]+knots[0]):.4f}, "
        f"scale {scale:.4f})")
    return field


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary-file",
                    help="required for --start seed (default)")
    ap.add_argument("--seed-index", type=int, default=None)
    ap.add_argument("--start", choices=("seed", "nae-nfp4"), default="seed",
                    help="seed = --boundary-file; nae-nfp4 = fresh rotating-"
                         "ellipse start for the L7 nfp-product test")
    ap.add_argument("--nfp", type=int, default=4,
                    help="fresh-start field periods (nae-nfp4 mode)")
    ap.add_argument("--aspect-target", type=float, default=9.5)
    ap.add_argument("--elongation-target", type=float, default=2.5)
    ap.add_argument("--axis-r", type=float, default=0.08)
    ap.add_argument("--axis-z", type=float, default=0.2,
                    help="axis torsion knob; hard-capped at 0.25 (wiki/05)")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--psi", type=float, default=3e-2)
    ap.add_argument("--maxiter", type=int, default=2)
    ap.add_argument("--floor-delta", type=float, default=0.02,
                    help="BScaleLength floor = (1+delta)*min(current); "
                         "negative releases the floor below the start")
    ap.add_argument("--og-grid-factor", type=int, default=4,
                    help="Omnigenity LCFS grid resolution = factor*eq.M/N "
                         "(Proxima default 4; 3 shrinks the biggest compile)")
    ap.add_argument("--jac-chunk", type=int, default=16,
                    help="jacobian chunk size for solves and the main "
                         "objective (memory ~ m0 + m1*chunk)")
    ap.add_argument("--omnigenity", action="store_true")
    ap.add_argument("--omnigenity-weight", type=float, default=2.0,
                    help="weight of the Omnigenity term in the main "
                         "optimization")
    ap.add_argument("--well-from", metavar="BOUNDARY_JSON",
                    help="fit the omnigenity target field to THIS boundary's "
                         "solved equilibrium (a true QI donor), then optimize "
                         "the main boundary against it (implies --omnigenity)")
    ap.add_argument("--fit-well", action="store_true",
                    help="fit the OmnigenousField to the start equilibrium's "
                         "|B| (implies --omnigenity)")
    ap.add_argument("--fit-well-iters", type=int, default=15,
                    help="field-only lsq iterations for --fit-well (0 = "
                         "knots only)")
    ap.add_argument("--roundtrip-only", action="store_true")
    ap.add_argument("--smoke-test", action="store_true",
                    help="construct everything (eq, objectives, field), no "
                         "solve/optimize, exit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.fit_well or a.well_from:
        a.omnigenity = True

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.start == "nae-nfp4":
        b0 = fresh_nae_boundary(a.nfp, a.aspect_target, a.elongation_target,
                                a.axis_r, a.axis_z)
        print(f"fresh start: nfp={a.nfp} aspect~{a.aspect_target} "
              f"elong~{a.elongation_target} axis_z={min(a.axis_z, 0.25)}",
              flush=True)
    else:
        if not a.boundary_file:
            ap.error("--boundary-file is required with --start seed")
        b0 = load_boundary(a.boundary_file, a.seed_index)

    t0 = time.time()
    surf = to_desc_surface(b0)
    if a.roundtrip_only:
        b1 = from_desc_surface(surf)
        out.write_text(json.dumps({"boundary": b1, "roundtrip_only": True},
                                  indent=2))
        rc0, rc1 = np.asarray(b0["r_cos"]), np.asarray(b1["r_cos"])
        zs0, zs1 = np.asarray(b0["z_sin"]), np.asarray(b1["z_sin"])
        # compare on the overlapping canvas
        mr = min(rc0.shape[0], rc1.shape[0])
        c0, c1 = (rc0.shape[1] - 1) // 2, (rc1.shape[1] - 1) // 2
        nr = min(c0, c1)
        dr = np.abs(rc0[:mr, c0 - nr:c0 + nr + 1]
                    - rc1[:mr, c1 - nr:c1 + nr + 1]).max()
        dz = np.abs(zs0[:mr, c0 - nr:c0 + nr + 1]
                    - zs1[:mr, c1 - nr:c1 + nr + 1]).max()
        print(f"roundtrip max|dR|={dr:.3e} max|dZ|={dz:.3e} "
              f"[{time.time()-t0:.1f}s]")
        return 0

    from desc.equilibrium import Equilibrium
    from desc.grid import LinearGrid
    from desc.objectives import (AspectRatio, BScaleLength, CurrentDensity,
                                 Elongation, FixBoundaryR, FixCurrent,
                                 FixPressure, FixPsi, MirrorRatio,
                                 ObjectiveFunction, RotationalTransform)
    from desc.optimize import Optimizer

    print(f"building equilibrium M={a.M} N={a.N} psi={a.psi} "
          f"jac_chunk={a.jac_chunk}", flush=True)
    if a.smoke_test:
        eq = Equilibrium(Psi=a.psi, surface=surf, M=a.M, N=a.N,
                         check_orientation=False)
        print("smoke-test: skipping force solve", flush=True)
    else:
        eq = build_and_solve(surf, a.M, a.N, a.psi, chunk=a.jac_chunk)

    lcfs = LinearGrid(rho=np.array([1.0]), M=2 * eq.M, N=2 * eq.N,
                      NFP=eq.NFP, sym=False)
    # current min L_grad(B) on the LCFS -> objective floor
    data = eq.compute(["L_grad(B)"], grid=lcfs)
    L_min0 = float(np.min(data["L_grad(B)"]))
    floor = (1.0 + a.floor_delta) * L_min0
    iota_edge = float(eq.compute("iota")["iota"][-1])
    nfp = int(eq.NFP)
    iota_lb = 0.25 * nfp + 0.005
    print(f"start: min L_grad(B)={L_min0:.4f} -> floor {floor:.4f}; "
          f"edge iota (DESC sign) = {iota_edge:.4f}; |iota| lb {iota_lb}",
          flush=True)

    terms = [
        BScaleLength(eq=eq, bounds=(floor, np.inf), normalize=False,
                     normalize_target=False, grid=lcfs, weight=1.0),
        AspectRatio(eq=eq, bounds=(0, 10.0), weight=10.0),
        Elongation(eq=eq, bounds=(0, 5.0), weight=5.0),
        MirrorRatio(eq=eq, bounds=(0, 0.2), weight=5.0),
        RotationalTransform(
            eq=eq,
            bounds=((-np.inf, -iota_lb) if iota_edge < 0
                    else (iota_lb, np.inf)),
            weight=5.0),
    ]
    things = [eq]
    if a.omnigenity:
        from desc.magnetic_fields import OmnigenousField
        from desc.objectives import Omnigenity
        og_grid = LinearGrid(rho=np.array([1.0]),
                             M=a.og_grid_factor * eq.M,
                             N=a.og_grid_factor * eq.N,
                             NFP=nfp, sym=False)
        if a.well_from:
            donor_b = load_boundary(a.well_from, None)
            field = field_from_donor(
                donor_b, eq, og_grid, M=a.M, N=a.N, psi=a.psi,
                fit_iters=0 if a.smoke_test else a.fit_well_iters,
                jac_chunk=a.jac_chunk)
        elif a.fit_well:
            field = fit_omnigenous_field(
                eq, og_grid,
                fit_iters=0 if a.smoke_test else a.fit_well_iters)
        else:
            # B_lm is a flattened (L_B+1, M_B) array: monotonic spline knots
            # of ||B||(eta) per radial Chebyshev row. L_B=0, M_B=3 -> 3
            # knots, 0.85 -> 1.15 = mirror ratio 0.15 (inside P2's 0.2).
            # helicity=(0, nfp) = poloidally closed |B| contours (QI).
            field = OmnigenousField(L_B=0, M_B=3, L_x=1, M_x=1, N_x=1,
                                    NFP=nfp, helicity=(0, nfp),
                                    B_lm=np.array([0.85, 1.0, 1.15]))
        terms.append(Omnigenity(eq=eq, field=field, field_fixed=False,
                                eq_grid=og_grid, weight=a.omnigenity_weight))
        things.append(field)

    objective = ObjectiveFunction(tuple(terms), deriv_mode="batched",
                              jac_chunk_size=a.jac_chunk)
    idx_r00 = eq.surface.R_basis.get_idx(M=0, N=0)
    r_modes_to_fix = eq.surface.R_basis.modes[idx_r00]
    constraints = (
        FixBoundaryR(eq=eq, modes=r_modes_to_fix),
        CurrentDensity(eq=eq),
        FixPressure(eq=eq),
        FixCurrent(eq=eq),
        FixPsi(eq=eq),
    )
    if a.smoke_test:
        objective.build(verbose=0)
        for c in constraints:
            c.build(verbose=0)
        print(f"smoke-test OK: {len(terms)} objective terms + "
              f"{len(constraints)} constraints built "
              f"(things: {[type(t).__name__ for t in things]}) "
              f"[{time.time()-t0:.1f}s]", flush=True)
        return 0
    print(f"optimizing lsq-auglag maxiter={a.maxiter} "
          f"({len(terms)} objective terms)...", flush=True)
    t1 = time.time()
    results, _ = Optimizer("lsq-auglag").optimize(
        things=things, objective=objective, constraints=constraints,
        maxiter=a.maxiter, verbose=3, copy=True)
    eq2 = results[0]
    print(f"optimize done in {time.time()-t1:.1f}s; re-solving force balance",
          flush=True)
    eq2 = solve_force_chunked(eq2, a.jac_chunk)

    data2 = eq2.compute(["L_grad(B)"], grid=lcfs)
    L_min1 = float(np.min(data2["L_grad(B)"]))
    b1 = from_desc_surface(eq2.surface)
    out.write_text(json.dumps({
        "boundary": b1,
        "provenance": {
            "tool": "desc_propose.py", "desc": "0.17.1",
            "source": a.boundary_file, "seed_index": a.seed_index,
            "start": a.start,
            "nfp": nfp,
            "M": a.M, "N": a.N, "maxiter": a.maxiter,
            "floor": floor, "omnigenity": a.omnigenity,
            "omnigenity_weight": a.omnigenity_weight,
            "fit_well": a.fit_well, "well_from": a.well_from,
            "L_min_desc_before": L_min0, "L_min_desc_after": L_min1,
            "wall_s": round(time.time() - t0, 1),
            "note": "DESC-side numbers are proposals only; "
                    "pinned evaluator adjudicates."},
    }, indent=2))
    print(f"DESC-side min L_grad(B): {L_min0:.4f} -> {L_min1:.4f}  "
          f"({time.time()-t0:.1f}s total). Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
