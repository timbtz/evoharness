"""Differential-test driver for diffscore (Plan 2).

Runs INSIDE the pinned eval image (the host venv has no jax/constellaration):

  docker run --rm --cpus=2 -v <repo>:/work -w /work evoharness-stellar-eval \
      python -m experiments.diffscore.difftest --build-cache --n 200

Cache layout (runs/diffscore/oracle_cache/):
  index.jsonl            one row per boundary: key, stratum, oracle metrics,
                         normalized violations, feasibility, eval seconds
  <key>.npz              wout + boozer arrays the JAX modules consume
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
_ARCHIVE = _ROOT / "memory" / "stellar_p2" / "archive.jsonl"
_BANK = _ROOT / "tasks" / "stellar_p2" / "seed_bank.json"
_CACHE = _ROOT / "runs" / "diffscore" / "oracle_cache"

_FIVE = ["aspect_ratio", "edge_rotational_transform_over_n_field_periods",
         "qi", "edge_magnetic_mirror_ratio", "max_elongation"]

_WOUT_FIELDS = ["rmnc", "zmns", "xm", "xn", "xm_nyq", "xn_nyq", "raxis_cc",
                "zaxis_cs", "iotaf", "bmnc", "gmnc", "bsupumnc", "bsupvmnc",
                "aspect", "Aminor_p", "volume_p",
                "normalized_toroidal_flux_full_grid_mesh",
                "normalized_toroidal_flux_half_grid_mesh"]


def stratum_of(feas: float) -> str:
    if feas <= 1e-2:
        return "feasible"
    if feas <= 2e-2:
        return "camped"
    if feas <= 2e-1:
        return "near"
    return "far"


def load_corpus(n: int, seed: int = 0) -> list[dict]:
    """Stratified boundary sample: all bank seeds + archive rows spread over
    feasible / camped / near / far strata (round-robin, deterministic)."""
    bank = json.loads(_BANK.read_text())["seeds"]
    corpus = [{"key": f"bank-{i}", "stratum": "bank", "boundary": s["boundary"]}
              for i, s in enumerate(bank)]
    strata: dict[str, list[dict]] = {"feasible": [], "camped": [], "near": [],
                                     "far": []}
    for line in _ARCHIVE.read_text().splitlines():
        r = json.loads(line)
        strata[stratum_of(r["feasibility"])].append(r)
    rng = np.random.default_rng(seed)
    for rows in strata.values():
        rng.shuffle(rows)  # type: ignore[arg-type]
    order = ["feasible", "camped", "near", "far"]
    i = 0
    while len(corpus) < n and any(strata[s] for s in order):
        s = order[i % len(order)]
        if strata[s]:
            r = strata[s].pop()
            corpus.append({"key": r["key"], "stratum": s,
                           "boundary": r["boundary"],
                           "archived_feasibility": r["feasibility"],
                           "archived_fidelity": r["fidelity"]})
        i += 1
    return corpus[:n]


def run_oracle(boundary_dict: dict, fidelity: str = "very_low_fidelity"):
    """Pinned forward model + boozer; returns (metrics_dict, arrays_dict)."""
    from constellaration import forward_model as fmod
    from constellaration import problems
    from constellaration.boozer import boozer as boozer_module
    from constellaration.geometry import surface_rz_fourier as srf
    from constellaration.mhd import vmec_settings as vs

    p2 = problems.SimpleToBuildQIStellarator()
    b = srf.SurfaceRZFourier.model_validate(boundary_dict)
    settings = fmod.ConstellarationSettings(
        vmec_preset_settings=vs.VmecPresetSettings(fidelity=fidelity),
        turbulent_settings=None)
    m, eq = fmod.forward_model(b, settings=settings)
    arrays = {f: np.asarray(getattr(eq, f)) for f in _WOUT_FIELDS}
    arrays.update(ns=eq.ns, mpol=eq.mpol, ntor=eq.ntor, nfp=eq.nfp,
                  lasym=eq.lasym)
    bs = boozer_module.create_boozer_settings_from_equilibrium_resolution(
        mhd_equilibrium=eq,
        settings=boozer_module.BoozerPresetSettings(
            normalized_toroidal_flux=[1.0]))
    bz = boozer_module.run_boozer(equilibrium=eq, settings=bs)
    arrays.update(
        bmnc_b=np.asarray(bz.bmnc_b), xm_b=np.asarray(bz.xm_b),
        xn_b=np.asarray(bz.xn_b), boozer_iota=np.asarray(bz.iota),
        s_in=np.asarray(bz.normalized_toroidal_flux),
        s_b=np.asarray(bz.boozer_normalized_toroidal_flux),
        bmnc_11=float(bz.bmnc[1, 1]))
    metrics = {k: (None if getattr(m, k) is None else float(getattr(m, k)))
               for k in _FIVE}
    metrics["minimum_normalized_magnetic_gradient_scale_length"] = float(
        m.minimum_normalized_magnetic_gradient_scale_length)
    metrics["feasibility"] = float(p2.compute_feasibility(m))
    metrics["violations"] = [float(v)
                             for v in p2._normalized_constraint_violations(m)]
    return metrics, arrays


def build_cache(n: int, fidelity: str = "very_low_fidelity") -> None:
    _CACHE.mkdir(parents=True, exist_ok=True)
    index_path = _CACHE / "index.jsonl"
    done = set()
    if index_path.exists():
        done = {json.loads(line)["key"]
                for line in index_path.read_text().splitlines()}
    # boundaries whose eval got the process killed (OOM on the shared box):
    # skip after 2 attempts so a retry loop can make progress past them
    inflight_path = _CACHE / "inflight.json"
    attempts = (json.loads(inflight_path.read_text())
                if inflight_path.exists() else {})
    corpus = load_corpus(n)
    todo = [c for c in corpus if c["key"] not in done]
    print(f"corpus {len(corpus)}, cached {len(corpus) - len(todo)}, "
          f"todo {len(todo)}", flush=True)
    for i, c in enumerate(todo):
        if attempts.get(c["key"], 0) >= 2:
            with index_path.open("a") as f:
                f.write(json.dumps({"key": c["key"], "stratum": c["stratum"],
                                    "error": "killed (OOM?) twice"}) + "\n")
            print(f"[{i + 1}/{len(todo)}] {c['key']} SKIP (killed twice)",
                  flush=True)
            continue
        attempts[c["key"]] = attempts.get(c["key"], 0) + 1
        inflight_path.write_text(json.dumps(attempts))
        t0 = time.time()
        try:
            metrics, arrays = run_oracle(c["boundary"], fidelity)
        except Exception as e:  # VMEC can fail on far-infeasible rows
            row = {"key": c["key"], "stratum": c["stratum"], "error": repr(e)}
            with index_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i + 1}/{len(todo)}] {c['key']} ERROR {e!r}", flush=True)
            continue
        np.savez_compressed(_CACHE / f"{c['key']}.npz", **arrays)
        row = {"key": c["key"], "stratum": c["stratum"], "fidelity": fidelity,
               "secs": round(time.time() - t0, 2),
               "boundary": c["boundary"], **metrics}
        with index_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(todo)}] {c['key']} {c['stratum']} "
              f"feas={metrics['feasibility']:.4g} {row['secs']}s", flush=True)


def load_cache(strata: tuple[str, ...] = ()) -> list[dict]:
    """Rows from index.jsonl (with arrays lazily loadable via row['npz'])."""
    rows = []
    for line in (_CACHE / "index.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "error" in r or (strata and r["stratum"] not in strata):
            continue
        r["npz"] = _CACHE / f"{r['key']}.npz"
        rows.append(r)
    return rows


def norms(deltas: np.ndarray) -> dict:
    a = np.abs(np.asarray(deltas, dtype=float))
    return {"n": int(a.size), "L1": float(np.mean(a)),
            "L2": float(np.sqrt(np.mean(a**2))), "Linf": float(np.max(a))}


def report_margins(args) -> dict:
    from experiments.diffscore import margins_jax
    rows = load_cache()
    if args.n:
        rows = rows[:args.n]
    out = margins_jax.parity_report(rows, grad_check_every=args.grad_every)
    return out


def report_qi(args) -> dict:
    from experiments.diffscore import qi_jax
    rows = load_cache()
    if args.n:
        rows = rows[:args.n]
    return qi_jax.parity_report(rows, smoothing=args.smoothing,
                                grad_check_every=args.grad_every)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--margins", action="store_true")
    ap.add_argument("--qi", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--grad-every", type=int, default=0,
                    help="FD-check gradients on every k-th row (0 = off)")
    args = ap.parse_args()
    report = {}
    if args.build_cache:
        build_cache(args.n)
    if args.margins or args.all:
        report["margins"] = report_margins(args)
    if args.qi or args.all:
        report["qi"] = report_qi(args)
    if report:
        out = _ROOT / "runs" / "diffscore" / "report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
