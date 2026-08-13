"""Gradient-driven polish loop: a real d(metrics)/d(boundary) INSIDE the loop.

This is the run the campaign never did. The A/B used Plan-2's aspect gradient,
which is boundary-only and never touches the solver; the FD probe measured the
solver gradient but only took isolated one-off steps. This re-linearizes at
every step and descends, with the pinned evaluator adjudicating each one.

It does NOT need Plan 3's adjoint. The same gradient is obtained by finite
differences at ~2 solves per coefficient (~40 per step, ~10 min) instead of the
adjoint's ~2-3 solve-equivalents. The adjoint would make this ~20x cheaper —
which is what would let it live inside the LLM loop — but the science question
("does gradient descent on this landscape buy honest score, and does it
compound?") is answerable now.

What it optimizes, and why not qi:

  honest score = p2 - 0.92 * max(0, feasibility - 0.002),   p2 = L_gradB / 20

  So the objective is L_gradB, and the constraints are what must not degrade.
  On the champion the ACTIVE constraint is aspect (violation 0.00208); qi sits
  just inside its bound (-0.00061). Descending qi, as the probe did, only buys
  headroom indirectly. This ascends L directly and projects the step orthogonal
  to whichever constraint gradients are active, so the margin is held while the
  objective moves.

Estimators (both validated, see Plan-3-Preflight-Report.md):

  dL/dp   : analytic dL/d(wout) from lgradb_jax, contracted against finite
            differences of the WOUT ARRAYS. Never finite-difference L itself —
            it is a min over a grid and its argmin switches under 1e-4 steps.
  dqi/dp  : analytic dqi/d(bmnc_b) from qi_jax, contracted the same way.
            Verified against a direct oracle FD to 0.02%.
  daspect : exact and analytic, free, no solve.

Trust region: 3e-5 max-coefficient. MEASURED, not guessed — predicted-vs-actual
ratio is 0.99 at 3e-5 and -0.05 at 1e-4. Steps that fail verification shrink it.

Resumable: state.json after every step.

  docker run --rm --cpus 2 --memory 5g --user $(id -u):$(id -g) -e HOME=/tmp \
      -v $PWD:/work -w /work evoharness-stellar-eval \
      python -u experiments/qi_polish.py --steps 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "runs" / "qi_polish"
ASPECT_BOUND = 10.0
MARGIN_TARGET, MARGIN_SLOPE = 0.002, 0.92
TRUST = 3e-5          # measured trust region (max-coefficient)


def honest_of(m: dict) -> float:
    p2 = m["minimum_normalized_magnetic_gradient_scale_length"] / 20.0
    return p2 - MARGIN_SLOPE * max(0.0, m["feasibility"] - MARGIN_TARGET)


def gradients(b0: dict, arrays: dict, coeffs, h: float, fidelity: str,
              log=print):
    """One re-linearization: returns (dL/dp, dqi/dp, n_solves).

    Both are contractions of an analytic downstream gradient (evaluated once at
    the base point) against finite differences of the solver's output arrays.
    """
    import jax
    import jax.numpy as jnp
    from experiments.diffscore import lgradb_jax as lg, qi_jax
    from experiments.diffscore.difftest import run_oracle
    from experiments.fd_qi_probe import perturbed

    b = arrays
    ns, mpol, ntor = int(b["ns"]), int(b["mpol"]), int(b["ntor"])
    nfp, lasym = int(b["nfp"]), bool(b["lasym"])

    L_ARR = ["rmnc", "zmns", "gmnc", "bmnc", "bsupumnc", "bsupvmnc"]

    def f_L(*arrs):
        d = dict(zip(L_ARR, arrs))
        return lg.min_normalized_l_grad_b(
            d["rmnc"], d["zmns"], d["gmnc"], d["bmnc"], d["bsupumnc"],
            d["bsupvmnc"], b["xm"], b["xn"], b["xm_nyq"], b["xn_nyq"], ns,
            float(b["Aminor_p"]), mpol, ntor, nfp, lasym)

    _, gL = jax.value_and_grad(f_L, argnums=tuple(range(len(L_ARR))))(
        *[jnp.asarray(b[k]) for k in L_ARR])
    gL = {k: np.asarray(g) for k, g in zip(L_ARR, gL)}

    def f_qi(bmnc_b, iota):
        return jnp.log10(qi_jax.qi_metric(
            bmnc_b, b["xm_b"], b["xn_b"], iota, b["s_in"], b["s_b"],
            nfp, float(b["bmnc_11"])))

    _, (gq_b, gq_i) = jax.value_and_grad(f_qi, argnums=(0, 1))(
        jnp.asarray(b["bmnc_b"]), jnp.asarray(b["boozer_iota"]))
    gq_b, gq_i = np.asarray(gq_b), np.asarray(gq_i)

    rc = np.asarray(b0["r_cos"], float)
    zs = np.asarray(b0["z_sin"], float)
    dL_rc, dL_zs = np.zeros_like(rc), np.zeros_like(zs)
    dq_rc, dq_zs = np.zeros_like(rc), np.zeros_like(zs)
    solves = 0
    for n, (tag, i, j, _v) in enumerate(coeffs):
        _, ap = run_oracle(perturbed(b0, tag, i, j, +h), fidelity)
        _, am = run_oracle(perturbed(b0, tag, i, j, -h), fidelity)
        solves += 2
        if any(ap[k].shape != b[k].shape or am[k].shape != b[k].shape
               for k in L_ARR + ["bmnc_b", "boozer_iota"]):
            log(f"    col {tag}[{i},{j}] dropped (solver grid changed)")
            continue
        dl = sum(float(np.sum(gL[k] * (ap[k] - am[k]) / (2 * h))) for k in L_ARR)
        dq = float(np.sum(gq_b * (ap["bmnc_b"] - am["bmnc_b"]) / (2 * h))) \
            + float(np.sum(gq_i * (ap["boozer_iota"] - am["boozer_iota"]) / (2 * h)))
        (dL_rc if tag == "r_cos" else dL_zs)[i, j] = dl
        (dq_rc if tag == "r_cos" else dq_zs)[i, j] = dq
        if n % 5 == 4:
            jax.clear_caches()
    return (np.concatenate([dL_rc.ravel(), dL_zs.ravel()]),
            np.concatenate([dq_rc.ravel(), dq_zs.ravel()]), solves)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--h", type=float, default=3e-5)
    ap.add_argument("--trust", type=float, default=TRUST)
    ap.add_argument("--fidelity", default="very_low_fidelity")
    ap.add_argument("--boundary-file",
                    help="JSON boundary (or object containing a boundary key)")
    ap.add_argument("--out", help="isolated output directory; required for parallel studies")
    ap.add_argument("--fresh", action="store_true",
                    help="start a new trajectory even if state.json exists")
    a = ap.parse_args()

    from experiments.diffscore.difftest import run_oracle
    from experiments.fd_qi_probe import aspect_grad, load_cases, pick_coeffs

    out = Path(a.out).resolve() if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    state_p = out / "state.json"
    if state_p.exists() and not a.fresh:
        st = json.loads(state_p.read_text())
        print(f"resuming at step {len(st['trajectory']) - 1}", flush=True)
    else:
        if a.boundary_file:
            payload = json.loads(Path(a.boundary_file).read_text())
            boundary = payload.get("boundary", payload)
        else:
            boundary = load_cases(["champion"])[0]["boundary"]
        case = {"boundary": boundary}
        m0, _ = run_oracle(case["boundary"], a.fidelity)
        st = {"boundary": case["boundary"], "trust": a.trust, "solves": 1,
              "trajectory": [{"step": 0, "honest": honest_of(m0),
                              "feasibility": m0["feasibility"],
                              "L": m0["minimum_normalized_magnetic_gradient_scale_length"],
                              "qi": m0["qi"], "aspect": m0["aspect_ratio"]}]}
        state_p.write_text(json.dumps(st, indent=2))
    h0 = st["trajectory"][0]["honest"]
    print(f"start honest={h0:.6f}", flush=True)

    for step in range(len(st["trajectory"]), a.steps + 1):
        t0 = time.time()
        b0 = st["boundary"]
        m_cur, arrays = run_oracle(b0, a.fidelity)
        st["solves"] += 1
        coeffs = pick_coeffs(b0, a.k)
        gL, gq, ns_ = gradients(b0, arrays, coeffs, a.h, a.fidelity)
        st["solves"] += ns_

        asp, asp_v, ga_rc, ga_zs = aspect_grad(b0)
        ga = np.concatenate([ga_rc.ravel(), ga_zs.ravel()])

        # Ascend the composite that acceptance actually scores.  The old loop
        # projected away all active-margin motion, even though honest_score has
        # an explicit exchange rate between L and the worst violation.
        d = gL / 20.0
        active = []
        violations = np.asarray(m_cur["violations"], float)
        worst = int(np.argmax(violations))
        if violations[worst] > MARGIN_TARGET and worst == 0:
            d -= MARGIN_SLOPE * ga
            active.append("aspect")
        elif violations[worst] > MARGIN_TARGET and worst == 2:
            # qi violation normalizes log10(qi) by abs(-4).
            d -= MARGIN_SLOPE * gq / 4.0
            active.append("qi")
        elif violations[worst] > MARGIN_TARGET:
            # We do not have complete iota/mirror/elongation boundary gradients.
            # Conservatively hold the trustworthy active margins in this case.
            rows = [v for v, on in ((ga, asp_v >= MARGIN_TARGET * 0.9),
                                     (gq, violations[2] > -2e-3))
                    if on and np.linalg.norm(v) > 0]
            if rows:
                A = np.vstack(rows)
                lam, *_ = np.linalg.lstsq(A @ A.T, A @ d, rcond=None)
                d -= A.T @ lam
                active.extend(["fallback-projection"])
        if not np.abs(d).max():
            print("zero step direction — stopping", flush=True)
            break
        step_vec = d / np.abs(d).max() * st["trust"]

        rc = np.asarray(b0["r_cos"], float)
        zs = np.asarray(b0["z_sin"], float)
        cand = json.loads(json.dumps(b0))
        cand["r_cos"] = (rc + step_vec[:rc.size].reshape(rc.shape)).tolist()
        cand["z_sin"] = (zs + step_vec[rc.size:].reshape(zs.shape)).tolist()
        pred_dL = float(gL @ step_vec)

        try:
            m, _ = run_oracle(cand, a.fidelity)
            st["solves"] += 1
            err = None
        except Exception as e:
            m, err = None, f"{type(e).__name__}: {e}"[:100]

        # The incumbent's score, NOT the last record's: after a rejection
        # trajectory[-1] holds the REJECTED candidate, whose honest score is
        # lower, and comparing against it accepts a regression on the next step
        # (measured: step 8 replaced the step-6 incumbent with one 1.4e-4 worse,
        # and st["boundary"] is the only copy). st["boundary"] always comes from
        # step 0 or the last accepted step, so the baseline must too.
        _inc = [r for r in st["trajectory"]
                if r.get("accepted") or r["step"] == 0]
        cur_h = _inc[-1]["honest"]
        if m is None:
            rec = {"step": step, "error": err, "trust": st["trust"],
                   "active": active}
            st["trust"] /= 3.0
            print(f"  step {step}: SOLVE FAILED ({err}) -> trust {st['trust']:.1e}",
                  flush=True)
        else:
            h = honest_of(m)
            rec = {"step": step, "honest": h, "delta": h - cur_h,
                   "predicted_dL": pred_dL,
                   "actual_dL": m["minimum_normalized_magnetic_gradient_scale_length"]
                   - m_cur["minimum_normalized_magnetic_gradient_scale_length"],
                   "feasibility": m["feasibility"], "qi": m["qi"],
                   "aspect": m["aspect_ratio"], "trust": st["trust"],
                   "active": active, "wall_s": round(time.time() - t0, 1)}
            if h > cur_h:
                st["boundary"] = cand
                rec["accepted"] = True
                # separate copy: state.json's boundary is overwritten in place,
                # so a regression (or a bad resume) would otherwise lose the best
                (out / "best_boundary.json").write_text(json.dumps(
                    {"honest": h, "step": step, "feasibility": m["feasibility"],
                     "boundary": cand}, indent=2))
            else:
                st["trust"] /= 3.0
                rec["accepted"] = False
            print(f"  step {step}: honest={h:.6f} (d={h-cur_h:+.6f}) "
                  f"predL={pred_dL:+.4f} actualL={rec['actual_dL']:+.4f} "
                  f"feas={m['feasibility']:.5f} active={active} "
                  f"{'ACCEPT' if rec['accepted'] else 'reject -> trust %.1e' % st['trust']}"
                  f"  [{rec['wall_s']/60:.1f}min, {st['solves']} solves]",
                  flush=True)
        st["trajectory"].append(rec)
        state_p.write_text(json.dumps(st, indent=2))
        if st["trust"] < 1e-7:
            print("trust region collapsed — stopping", flush=True)
            break

    good = [t for t in st["trajectory"] if "honest" in t]
    print(f"\n=== polish loop: {len(good)-1} steps, {st['solves']} solves ===")
    print(f"start honest {h0:.6f} -> best {max(t['honest'] for t in good):.6f} "
          f"(delta {max(t['honest'] for t in good) - h0:+.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
