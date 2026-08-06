"""Does the gradient tool fail because gradients don't help, or because it is
used 100x outside the range where its gradient is valid?

Two measurements point at the same answer and this one tests it directly:

* The A/B (p1): the treated arm lost by 0.002 honest and produced the only
  boundaries that failed the authoritative re-score, all from `margin_step`.
* The pre-flight (Plan-3-Preflight-Report.md 4b): the P2 objective's argmin
  switches under 1e-4 max-coefficient steps; at 1e-5 the analytic gradient
  agrees with its own secant to 2.2%.

And the writers in the treated arm chose `cap` = 2e-3..4e-3 — 200-400x above
that regime.

So: walk the SAME boundary toward the SAME aspect-margin target with the same
tool, varying only the per-step cap, and score every step at the pinned
evaluator. If the tool is fine and the step size was the problem, small caps
should hold or improve the honest score while large caps degrade it or leave
the manifold. If small caps fail too, the tool itself is the problem and the
step-size story is wrong.

Uses the SHIPPED `_aspect_walk` out of tasks/stellar_p2/task.py — the exact code
the candidates called, not a re-implementation.

  docker run --rm --cpus 2 --memory 3g --user $(id -u):$(id -g) -e HOME=/tmp \
      -v $PWD:/work -w /work evoharness-stellar-eval \
      python experiments/cap_sweep.py --caps 3e-3,1e-3,1e-4,1e-5 --steps 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "runs" / "cap_sweep"
MARGIN_TARGET, MARGIN_SLOPE = 0.002, 0.92


def honest(metrics: dict) -> float:
    p2 = metrics["minimum_normalized_magnetic_gradient_scale_length"] / 20.0
    return p2 - MARGIN_SLOPE * max(0.0, metrics["feasibility"] - MARGIN_TARGET)


def champion() -> dict:
    """Best honest feasible boundary in the archive — the same starting point
    the campaign's runs resume from."""
    rows = [json.loads(x) for x in
            (_ROOT / "memory" / "stellar_p2" / "archive.jsonl")
            .read_text().splitlines() if x.strip()]
    ok = [r for r in rows if r.get("boundary") and r.get("p2")
          and (r.get("feasibility") if r.get("feasibility") is not None else 9) <= 1e-2]
    return max(ok, key=lambda r: r["p2"]
               - MARGIN_SLOPE * max(0.0, r["feasibility"] - MARGIN_TARGET))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="3e-3,1e-3,1e-4,1e-5")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--target", type=float, default=0.002)
    ap.add_argument("--fidelity", default="very_low_fidelity")
    a = ap.parse_args()

    # metrics only — difftest.run_oracle additionally runs the Boozer
    # transform to produce JAX arrays, and THAT is what does not fit in a 2 GB
    # container while the campaign holds the box (measured: rss is 54 MB right
    # up to the oracle call, then the container is OOM-killed). The forward
    # model already returns qi, so the boozer pass is pure overhead here.
    def run_oracle(boundary_dict, fidelity):
        from constellaration import forward_model as fmod, problems
        from constellaration.geometry import surface_rz_fourier as srf
        from constellaration.mhd import vmec_settings as vs
        b = srf.SurfaceRZFourier.model_validate(boundary_dict)
        m, _eq = fmod.forward_model(b, settings=fmod.ConstellarationSettings(
            vmec_preset_settings=vs.VmecPresetSettings(fidelity=fidelity),
            turbulent_settings=None))
        p2 = problems.SimpleToBuildQIStellarator()
        return {
            "aspect_ratio": float(m.aspect_ratio),
            "qi": None if m.qi is None else float(m.qi),
            "minimum_normalized_magnetic_gradient_scale_length": float(
                m.minimum_normalized_magnetic_gradient_scale_length),
            "feasibility": float(p2.compute_feasibility(m)),
        }, None

    # the SHIPPED walk, exec'd into the task module by task.py itself
    import tasks.stellar_p2.task as T

    OUT.mkdir(parents=True, exist_ok=True)
    base = champion()
    b0 = base["boundary"]
    m0, _ = run_oracle(b0, a.fidelity)
    h0 = honest(m0)
    print(f"start: honest={h0:.6f} p2={m0['minimum_normalized_magnetic_gradient_scale_length']/20:.6f} "
          f"feas={m0['feasibility']:.5f} aspect={m0['aspect_ratio']:.5f}", flush=True)

    results = []
    for cap in [float(c) for c in a.caps.split(",")]:
        b = json.loads(json.dumps(b0))
        traj = [{"step": 0, "honest": h0, "feasibility": m0["feasibility"],
                 "aspect": m0["aspect_ratio"]}]
        print(f"\n--- cap={cap:g}", flush=True)
        for i in range(a.steps):
            rc, zs = T._aspect_walk(b["r_cos"], b["z_sin"],
                                    b["n_field_periods"], a.target, cap)
            b = json.loads(json.dumps(b))
            b["r_cos"], b["z_sin"] = np.asarray(rc).tolist(), np.asarray(zs).tolist()
            try:
                m, _ = run_oracle(b, a.fidelity)
            except Exception as e:
                traj.append({"step": i + 1, "error": f"{type(e).__name__}: {e}"[:120]})
                print(f"  step {i+1}: SOLVER/VALIDATION FAILURE "
                      f"{type(e).__name__}: {str(e)[:70]}", flush=True)
                break
            h = honest(m)
            traj.append({"step": i + 1, "honest": h,
                         "feasibility": m["feasibility"],
                         "aspect": m["aspect_ratio"],
                         "qi": m.get("qi")})
            print(f"  step {i+1}: honest={h:+.6f} (d={h-h0:+.6f}) "
                  f"feas={m['feasibility']:.5f} aspect={m['aspect_ratio']:.5f} "
                  f"qi={m.get('qi')}", flush=True)
        good = [t for t in traj if "honest" in t]
        results.append({"cap": cap, "trajectory": traj,
                        "best_honest": max(t["honest"] for t in good),
                        "best_delta": max(t["honest"] for t in good) - h0,
                        "failed_at": next((t["step"] for t in traj
                                           if "error" in t), None)})
    (OUT / "cap_sweep.json").write_text(
        json.dumps({"start_honest": h0, "results": results}, indent=2))

    print("\n=== summary (start honest = %.6f) ===" % h0)
    for r in results:
        fail = f"  FAILED at step {r['failed_at']}" if r["failed_at"] else ""
        print(f"cap={r['cap']:<8g} best honest={r['best_honest']:.6f} "
              f"(delta {r['best_delta']:+.6f}){fail}")
    print(f"\nwrote {OUT / 'cap_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
