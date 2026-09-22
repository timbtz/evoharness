"""Parity + gradient test for mirror_jax vs the pinned oracle cache.

Runs inside the eval image, uses only cached arrays (no solves):

  docker run --rm --cpus 1 --memory 2g --user $(id -u):$(id -g) -e HOME=/tmp \
      -v $PWD:/work -w /work evoharness-stellar-eval \
      python -m evoharness.tasks.stellar_p2.physics.metrics.mirror_parity
"""
from __future__ import annotations

import json
import sys

import numpy as np

from evoharness.paths import RUNS_DIR

_CACHE = RUNS_DIR / "diffscore" / "oracle_cache"


def main() -> int:
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    import jax
    import jax.numpy as jnp
    from evoharness.tasks.stellar_p2.physics.metrics import mirror_jax as mj

    rows = [json.loads(x) for x in
            (_CACHE / "index.jsonl").read_text().splitlines()]
    # mixed strata: bank + first few of each stratum with a cached npz
    picked, seen = [], set()
    for r in rows:
        if r["stratum"] not in seen or len(picked) < 10:
            if (_CACHE / f"{r['key']}.npz").exists():
                picked.append(r)
                seen.add(r["stratum"])
        if len(picked) >= 12:
            break

    worst = 0.0
    for r in picked:
        z = np.load(_CACHE / f"{r['key']}.npz")
        val = float(mj.edge_mirror_ratio(
            jnp.asarray(z["bmnc"]), jnp.asarray(z["xm_nyq"]),
            jnp.asarray(z["xn_nyq"]),
            jnp.asarray(z["normalized_toroidal_flux_half_grid_mesh"]),
            int(z["mpol"]), int(z["ntor"]), int(z["nfp"]), bool(z["lasym"])))
        ref = r["edge_magnetic_mirror_ratio"]
        err = abs(val - ref)
        worst = max(worst, err)
        print(f"{r['key']:24} {r['stratum']:9} jax={val:.12f} "
              f"oracle={ref:.12f} |d|={err:.2e}", flush=True)
    print(f"METRIC PARITY worst |delta| = {worst:.3e}")

    # gradient check: autodiff vs central FD of bmnc perturbations
    rng = np.random.default_rng(0)
    gworst = 0.0
    for r in picked[:3]:
        z = np.load(_CACHE / f"{r['key']}.npz")
        args = (jnp.asarray(z["xm_nyq"]), jnp.asarray(z["xn_nyq"]),
                jnp.asarray(z["normalized_toroidal_flux_half_grid_mesh"]),
                int(z["mpol"]), int(z["ntor"]), int(z["nfp"]),
                bool(z["lasym"]))

        def f(b):
            return mj.edge_mirror_ratio(b, *args)

        b0 = jnp.asarray(z["bmnc"])
        g = np.asarray(jax.grad(f)(b0))
        for _ in range(3):
            v = rng.standard_normal(b0.shape)
            v /= np.linalg.norm(v)
            h = 1e-6 * float(np.abs(z["bmnc"]).max())
            fp = float(f(b0 + h * v))
            fm = float(f(b0 - h * v))
            fd = (fp - fm) / (2 * h)
            an = float(np.sum(g * v))
            rel = abs(fd - an) / max(abs(fd), abs(an), 1e-30)
            gworst = max(gworst, rel)
            print(f"{r['key']:24} dir-deriv analytic={an:+.6e} "
                  f"fd={fd:+.6e} rel={rel:.2e}", flush=True)
    print(f"GRADIENT worst rel err = {gworst:.3e}")
    ok = worst <= 1e-10 and gworst <= 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
