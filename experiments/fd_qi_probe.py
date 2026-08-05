"""FD probe: what would d(qi)/d(boundary) buy us? — the Plan-3 go/no-go.

Plan 2 left the score differentiable everywhere EXCEPT across the solver: it has
d(qi)/d(bmnc_b) analytically, but not d(bmnc_b)/d(boundary). Plan 3 proposes to
build that link as an IFT adjoint (~2-3 solve-equivalents per gradient). This
probe gets the SAME gradient by finite differences (~2 solves per coefficient,
i.e. ~100x more expensive) on a handful of boundaries — enough to answer "is the
direction worth anything?" BEFORE any adjoint is written.

Estimator (deliberately not a plain FD of qi):

    d log10(qi)/dp_k  =  <dqi/d(bmnc_b), d(bmnc_b)/dp_k>
                       + <dqi/d(iota),   d(iota)/dp_k>          / (qi ln10)

The downstream factors are Plan-2's validated JAX gradients evaluated ONCE at
the base point; only the solver link is finite-differenced. This matters: qi.py
is non-smooth (argsort squash/stretch, bounce-point birth), and Plan 2 measured
plain FD of qi disagreeing with the a.e. gradient by up to 3e-3 relative across
kinks. The Boozer arrays themselves are smooth in the boundary, so FD'ing only
them keeps the kink noise out of the stencil.

Guards that matter on this task:
* Only ALREADY-NONZERO coefficients are perturbed. The vlf preset sets
  mpol/ntor from the boundary's largest nonzero modes, so waking a zero
  coefficient changes the solver grid and the Boozer array shapes — the column
  would be measuring a grid change, not a derivative. Shape mismatch => the
  column is dropped and reported, never silently used.
* Central differences, all-cold solves (never mixing warm and cold in one
  stencil — Plan-1 report), at two step sizes on a subset so truncation/noise
  is bounded rather than assumed.
* Every proposed step is verified by a fresh pinned forward call. The probe
  never claims an improvement it did not re-score.

Runs INSIDE the pinned eval image (needs constellaration + jax):

  docker run --rm --memory 5g -v <repo>:/work -w /work evoharness-stellar-eval \
      python experiments/fd_qi_probe.py --cases champion,davidkh --k 20

Results stream to runs/fd_probe/<case>/ (resumable: completed columns are
skipped on re-run).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "runs" / "fd_probe"
ASPECT_BOUND = 10.0


# ---------------------------------------------------------------------------
# boundaries under test
# ---------------------------------------------------------------------------

def load_cases(names: list[str]) -> list[dict]:
    """champion = the best boundary the gradient-free deep runs produced;
    davidkh = the public #1 seed; archive:<n> = n archive rows by feasibility."""
    cases: list[dict] = []
    for name in names:
        if name == "davidkh":
            bank = json.loads(
                (_ROOT / "tasks" / "stellar_p2" / "seed_bank.json").read_text())
            cases.append({"name": "davidkh",
                          "boundary": bank["seeds"][0]["boundary"]})
        elif name == "champion":
            rows = [json.loads(x) for x in
                    (_ROOT / "memory" / "stellar_p2" / "archive.jsonl")
                    .read_text().splitlines() if x.strip()]
            # archive schema: {"p2", "shaped", "feasibility", "fidelity", ...}.
            # Rank by the HONEST score (p2 discounted to the 0.002 margin target
            # at the official 0.92 slope) so the probe starts from the best
            # honest boundary, not the one that camped the tolerance hardest.
            ok = [r for r in rows
                  if r.get("boundary") and r.get("p2")
                  and (r.get("feasibility") if r.get("feasibility") is not None
                       else 9) <= 1e-2]
            if not ok:
                raise SystemExit("no feasible archive row to use as champion")

            def honest(r):
                return r["p2"] - 0.92 * max(0.0, r["feasibility"] - 0.002)

            best = max(ok, key=honest)
            cases.append({"name": "champion", "boundary": best["boundary"],
                          "archive_p2": best.get("p2"),
                          "archive_honest": round(honest(best), 6),
                          "archive_feasibility": best.get("feasibility"),
                          "archive_key": best.get("key")})
        else:
            raise SystemExit(f"unknown case {name!r}")
    return cases


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def free_mask(rc: np.ndarray, zs: np.ndarray):
    """Coefficients a step may touch (stellarator symmetry pins the rest)."""
    ntor = (rc.shape[1] - 1) // 2
    mr, mz = np.ones_like(rc), np.ones_like(zs)
    mr[0, :ntor] = 0.0
    mz[0, :ntor + 1] = 0.0
    return mr, mz


def pick_coeffs(boundary: dict, k: int) -> list[tuple[str, int, int, float]]:
    """The k largest free NONZERO coefficients — see the grid-change guard."""
    rc = np.asarray(boundary["r_cos"], float)
    zs = np.asarray(boundary["z_sin"], float)
    mr, mz = free_mask(rc, zs)
    items = []
    for arr, mask, tag in ((rc, mr, "r_cos"), (zs, mz, "z_sin")):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = float(arr[i, j])
                if mask[i, j] and v != 0.0:
                    items.append((tag, i, j, v))
    items.sort(key=lambda t: -abs(t[3]))
    return items[:k]


def perturbed(boundary: dict, tag: str, i: int, j: int, dx: float) -> dict:
    b = json.loads(json.dumps(boundary))
    a = np.asarray(b[tag], float)
    a[i, j] += dx
    b[tag] = a.tolist()
    return b


def downstream_grads(arrays: dict):
    """Plan-2 analytic d(log10 qi)/d(bmnc_b) and /d(boozer_iota) at the base
    point. Returns (log10_qi, g_bmnc, g_iota)."""
    import jax
    import jax.numpy as jnp
    from experiments.diffscore import qi_jax

    z = {k: arrays[k] for k in ("bmnc_b", "xm_b", "xn_b", "boozer_iota",
                                "s_in", "s_b", "nfp", "bmnc_11")}

    def f(bmnc_b, iota):
        q = qi_jax.qi_metric(bmnc_b, z["xm_b"], z["xn_b"], iota, z["s_in"],
                             z["s_b"], int(z["nfp"]), float(z["bmnc_11"]))
        return jnp.log10(q)

    val, (g_b, g_i) = jax.value_and_grad(f, argnums=(0, 1))(
        jnp.asarray(z["bmnc_b"]), jnp.asarray(z["boozer_iota"]))
    return float(val), np.asarray(g_b), np.asarray(g_i)


def aspect_grad(boundary: dict):
    """Exact analytic aspect + d(normalized violation)/d(coeff) (free)."""
    from experiments.diffscore import margins_jax as mj
    import jax
    import jax.numpy as jnp

    rc = jnp.asarray(boundary["r_cos"], float)
    zs = jnp.asarray(boundary["z_sin"], float)
    nfp = int(boundary["n_field_periods"])
    a = float(mj.aspect_ratio(rc, zs, nfp))
    g_rc, g_zs = jax.grad(lambda r, z: mj.aspect_ratio(r, z, nfp),
                          argnums=(0, 1))(rc, zs)
    mr, mz = free_mask(np.asarray(rc), np.asarray(zs))
    return (a, (a - ASPECT_BOUND) / ASPECT_BOUND,
            np.asarray(g_rc) * mr / ASPECT_BOUND,
            np.asarray(g_zs) * mz / ASPECT_BOUND)


def contract(g_b, g_i, base, plus, minus, h) -> float | None:
    """<downstream grad, central difference of the Boozer arrays>."""
    if (plus["bmnc_b"].shape != base["bmnc_b"].shape
            or minus["bmnc_b"].shape != base["bmnc_b"].shape
            or plus["boozer_iota"].shape != base["boozer_iota"].shape
            or minus["boozer_iota"].shape != base["boozer_iota"].shape):
        return None
    d_b = (plus["bmnc_b"] - minus["bmnc_b"]) / (2 * h)
    d_i = (plus["boozer_iota"] - minus["boozer_iota"]) / (2 * h)
    return float(np.sum(g_b * d_b) + np.sum(g_i * d_i))


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def probe_case(case: dict, k: int, h: float, h2: float, n_check: int,
               fidelity: str) -> dict:
    from experiments.diffscore.difftest import run_oracle

    name = case["name"]
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    cols_path = d / "columns.jsonl"
    done = {}
    if cols_path.exists():
        for line in cols_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["tag"], r["i"], r["j"], r["h"])] = r

    b0 = case["boundary"]
    t0 = time.time()
    print(f"[{name}] base solve ...", flush=True)
    m0, a0 = run_oracle(b0, fidelity)
    solves = 1
    log_qi0, g_b, g_i = downstream_grads(a0)
    asp, asp_v, ga_rc, ga_zs = aspect_grad(b0)
    print(f"[{name}] base: log10(qi)_jax={log_qi0:.6f} "
          f"qi_oracle={m0.get('qi'):.6g} aspect={asp:.6f} (v={asp_v:.5f}) "
          f"feas={m0['feasibility']:.5f} "
          f"L={m0['minimum_normalized_magnetic_gradient_scale_length']:.4f}",
          flush=True)

    coeffs = pick_coeffs(b0, k)
    grad = {}
    dropped = []
    for idx, (tag, i, j, v) in enumerate(coeffs):
        steps = [h] + ([h2] if idx < n_check else [])
        for hh in steps:
            key = (tag, i, j, hh)
            if key in done:
                rec = done[key]
            else:
                _, ap = run_oracle(perturbed(b0, tag, i, j, +hh), fidelity)
                _, am = run_oracle(perturbed(b0, tag, i, j, -hh), fidelity)
                solves += 2
                g = contract(g_b, g_i, a0, ap, am, hh)
                rec = {"tag": tag, "i": i, "j": j, "value": v, "h": hh,
                       "dlog10qi": g}
                with cols_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            if rec["dlog10qi"] is None:
                dropped.append(f"{tag}[{i},{j}] h={hh} (solver grid changed)")
            elif hh == h:
                grad[(tag, i, j)] = rec["dlog10qi"]
            print(f"[{name}] {idx + 1}/{len(coeffs)} {tag}[{i},{j}] "
                  f"h={hh:g} d log10(qi)={rec['dlog10qi']}", flush=True)
        if idx % 5 == 4:
            import jax
            jax.clear_caches()

    # step-size consistency on the checked subset
    consistency = []
    by_key: dict = {}
    for line in cols_path.read_text().splitlines():
        r = json.loads(line)
        by_key.setdefault((r["tag"], r["i"], r["j"]), {})[r["h"]] = r["dlog10qi"]
    for kk, vv in by_key.items():
        if h in vv and h2 in vv and vv[h] and vv[h2]:
            rel = abs(vv[h] - vv[h2]) / max(abs(vv[h]), 1e-30)
            consistency.append({"coeff": list(kk), f"h={h}": vv[h],
                                f"h={h2}": vv[h2], "rel_diff": rel})

    # ---- assemble the gradient over the full coefficient canvas ----
    rc = np.asarray(b0["r_cos"], float)
    zs = np.asarray(b0["z_sin"], float)
    gq_rc, gq_zs = np.zeros_like(rc), np.zeros_like(zs)
    for (tag, i, j), g in grad.items():
        (gq_rc if tag == "r_cos" else gq_zs)[i, j] = g

    result = {
        "case": name, "fidelity": fidelity, "k": len(coeffs), "h": h,
        "solves": solves, "wall_s": round(time.time() - t0, 1),
        "base": {"log10_qi_jax": log_qi0, "qi_oracle": m0.get("qi"),
                 "aspect": asp, "aspect_violation": asp_v,
                 "feasibility": m0["feasibility"],
                 "p2_L": m0["minimum_normalized_magnetic_gradient_scale_length"],
                 "violations": m0["violations"]},
        "grad_norm": float(np.linalg.norm(np.concatenate(
            [gq_rc.ravel(), gq_zs.ravel()]))),
        "dropped_columns": dropped,
        "step_consistency": consistency,
        "grad_r_cos": gq_rc.tolist(), "grad_z_sin": gq_zs.tolist(),
        "aspect_grad_r_cos": ga_rc.tolist(),
        "aspect_grad_z_sin": ga_zs.tolist(),
    }
    (d / "gradient.json").write_text(json.dumps(result, indent=2))
    return result


def descend(case: dict, res: dict, caps: list[float], fidelity: str) -> list[dict]:
    """Trust-region steps down the qi gradient, PROJECTED to hold the aspect
    margin, each verified by a fresh pinned forward call."""
    from experiments.diffscore.difftest import run_oracle

    b0 = case["boundary"]
    gq = np.concatenate([np.asarray(res["grad_r_cos"]).ravel(),
                         np.asarray(res["grad_z_sin"]).ravel()])
    ga = np.concatenate([np.asarray(res["aspect_grad_r_cos"]).ravel(),
                         np.asarray(res["aspect_grad_z_sin"]).ravel()])
    if np.linalg.norm(ga) > 0:
        gq = gq - ga * float(gq @ ga) / float(ga @ ga)  # hold aspect margin
    if not np.linalg.norm(gq):
        return []
    direction = -gq / np.abs(gq).max()  # max-coefficient normalized

    rc = np.asarray(b0["r_cos"], float)
    n_rc = rc.size
    rows = []
    for cap in caps:
        step = direction * cap
        b = json.loads(json.dumps(b0))
        b["r_cos"] = (rc + step[:n_rc].reshape(rc.shape)).tolist()
        b["z_sin"] = (np.asarray(b0["z_sin"], float)
                      + step[n_rc:].reshape(np.asarray(b0["z_sin"]).shape)).tolist()
        pred = float(gq @ step)  # predicted d log10(qi)
        try:
            m, _ = run_oracle(b, fidelity)
        except Exception as e:  # non-convergence must be loud, not silent
            rows.append({"cap": cap, "error": f"{type(e).__name__}: {e}"})
            print(f"  step cap={cap:g}: SOLVER FAILED {e}", flush=True)
            continue
        row = {"cap": cap, "predicted_dlog10qi": pred,
               "qi": m.get("qi"),
               "actual_dlog10qi": (math.log10(m["qi"]) - res["base"]["log10_qi_jax"])
               if m.get("qi") else None,
               "aspect": m.get("aspect_ratio"), "feasibility": m["feasibility"],
               "p2_L": m["minimum_normalized_magnetic_gradient_scale_length"],
               "violations": m["violations"]}
        rows.append(row)
        print(f"  step cap={cap:g}: pred dlog10qi={pred:+.4f} "
              f"actual={row['actual_dlog10qi']} feas={row['feasibility']:.5f} "
              f"L={row['p2_L']:.4f}", flush=True)
    (OUT / case["name"] / "descent.json").write_text(json.dumps(rows, indent=2))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="champion,davidkh")
    ap.add_argument("--k", type=int, default=20,
                    help="number of dominant free coefficients to probe")
    ap.add_argument("--h", type=float, default=3e-5)
    ap.add_argument("--h2", type=float, default=1e-5,
                    help="second step size, on the first --n-check coefficients")
    ap.add_argument("--n-check", type=int, default=5)
    ap.add_argument("--fidelity", default="very_low_fidelity")
    ap.add_argument("--caps", default="3e-4,1e-3,3e-3")
    ap.add_argument("--no-descent", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for case in load_cases(a.cases.split(",")):
        res = probe_case(case, a.k, a.h, a.h2, a.n_check, a.fidelity)
        print(f"[{case['name']}] gradient done: {res['solves']} solves, "
              f"{res['wall_s'] / 60:.1f} min, |g|={res['grad_norm']:.4g}, "
              f"dropped={len(res['dropped_columns'])}", flush=True)
        rows = [] if a.no_descent else descend(case, res,
                                               [float(x) for x in a.caps.split(",")],
                                               a.fidelity)
        summary.append({"case": case["name"], "solves": res["solves"],
                        "wall_min": round(res["wall_s"] / 60, 1),
                        "grad_norm": res["grad_norm"],
                        "base": res["base"], "descent": rows})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary, default=str)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
