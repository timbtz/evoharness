"""Homotopy scan between two boundary designs at a fixed fidelity.

Maps the objective/feasibility profile along the straight line in
scale-normalized, zero-padded Fourier space between two parents. Purpose
(wiki/03-leads.md L3a): the public leaderboard holds exactly two basins;
the barrier profile between them tells us whether recombination has a
shallow saddle to exploit or the basins are separated by a VMEC cliff.

Runs INSIDE the eval image (constellaration + diffscore deps):

  docker run --rm --cpus 1 --memory 3g --user $(id -u):$(id -g) -e HOME=/tmp \
      -v $PWD:/work -w /work evoharness-stellar-eval \
      python -m evoharness.tasks.stellar_p2.physics.homotopy \
        --a src/evoharness/tasks/stellar_p2/seed_bank.json:0 \
        --b src/evoharness/tasks/stellar_p2/seed_bank.json:1 \
        --points 21 --out .local/runs/homotopy-A-B/profile.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

def load_boundary(spec: str) -> dict:
    """path.json[:index] — bank file with :index, else a plain boundary JSON."""
    if ":" in spec and not spec.endswith(".json"):
        path, idx = spec.rsplit(":", 1)
        payload = json.loads(Path(path).read_text())
        if "seeds" in payload:
            return payload["seeds"][int(idx)]["boundary"]
        raise SystemExit(f"{path} has no seeds[] for index {idx}")
    payload = json.loads(Path(spec).read_text())
    return payload.get("boundary", payload)


def padded_pair(a: dict, b: dict):
    ra, za = np.array(a["r_cos"], float), np.array(a["z_sin"], float)
    rb, zb = np.array(b["r_cos"], float), np.array(b["z_sin"], float)
    M = max(ra.shape[0], rb.shape[0])
    N = max(ra.shape[1], rb.shape[1])

    def pad(rc, zs):
        prc = np.zeros((M, N)); pzs = np.zeros((M, N))
        m, n = rc.shape
        off = (N - n) // 2
        prc[:m, off:off + n] = rc
        pzs[:zs.shape[0], off:off + n] = zs
        return prc, pzs

    return pad(ra, za), pad(rb, zb), (M, N)


def _eval_child(cand: dict, fidelity: str, q) -> None:
    from evoharness.tasks.stellar_p2.physics.metrics.difftest import run_oracle
    m, _ = run_oracle(cand, fidelity)
    q.put({"feasibility": m["feasibility"],
           "L": m["minimum_normalized_magnetic_gradient_scale_length"],
           "qi": m["qi"], "aspect": m["aspect_ratio"],
           "violations": list(map(float, m["violations"]))})


def eval_point(cand: dict, fidelity: str, timeout: float = 300.0) -> dict:
    """One eval in its own process: an OOM-kill or hang loses the point, not
    the scan (measured 2026-08-16: the t=0.05 A->B interpolant blows past a
    4 GB cgroup inside the solver — SIGKILL, no traceback)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_eval_child, args=(cand, fidelity, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return {"error": f"timeout>{timeout:.0f}s"}
    if p.exitcode != 0:
        return {"error": f"child killed (exit {p.exitcode}, likely OOM/solver cliff)"}
    try:
        return q.get_nowait()
    except Exception:
        return {"error": "child exited 0 without result"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--points", type=int, default=21)
    ap.add_argument("--fidelity", default="very_low_fidelity")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ba, bb = load_boundary(args.a), load_boundary(args.b)
    if ba["n_field_periods"] != bb["n_field_periods"]:
        raise SystemExit("nfp mismatch — homotopy undefined")
    (ra, za), (rb, zb), _ = padded_pair(ba, bb)
    # scale-normalize each parent by its own R0 so the interpolation is in
    # the same dimensionless space the novelty metric uses; rescale to A's R0.
    R0a, R0b = ra[0, ra.shape[1] // 2], rb[0, rb.shape[1] // 2]
    ra, za = ra / R0a, za / R0a
    rb, zb = rb / R0b, zb / R0b

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.points):
        t = i / (args.points - 1)
        rc = ((1 - t) * ra + t * rb) * R0a
        zs = ((1 - t) * za + t * zb) * R0a
        cand = {"r_cos": rc.tolist(), "z_sin": zs.tolist(),
                "r_sin": None, "z_cos": None,
                "n_field_periods": int(ba["n_field_periods"]),
                "is_stellarator_symmetric": True}
        t0 = time.time()
        res = eval_point(cand, args.fidelity, args.timeout)
        res.update({"t": round(t, 4), "wall_s": round(time.time() - t0, 1)})
        rows.append(res)
        if "error" in res:
            print(f"t={t:.3f} SOLVE FAILED {res['error']}", flush=True)
        else:
            print(f"t={t:.3f} feas={res['feasibility']:.5f} L={res['L']:.4f} "
                  f"aspect={res['aspect']:.4f} [{res['wall_s']}s]", flush=True)
        out.write_text(json.dumps(
            {"a": args.a, "b": args.b, "fidelity": args.fidelity,
             "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
