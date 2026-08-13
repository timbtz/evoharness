"""First optimizer run with REAL solver gradients in the loop (STELLAR_FULL_GRAD).

What is new here, and why the previous campaign's null does not settle it: the
2026-08-05 A/B gave the writers `fm.margin_grad`/`fm.margin_step`, which are exact
but boundary-only — they steer ASPECT. Plan 2 had already measured that qi opposes
every geometric move at 1.5-3x strength, and on the champion boundary aspect is the
constraint sitting at the wall while qi is the one that actually resists. So that
campaign tested a gradient for the wrong constraint, and its "no detectable effect"
verdict is a statement about the aspect tool, not about gradient feedback.

This run hands the writers the two that bind: d(L-gradB)/d(boundary) — the objective
itself — and d(log10 qi)/d(boundary), via `fm.metric_grad`, plus `fm.grad_step` for
the projected step. Both come from `experiments/diffscore/boundary_grad.py`, the same
estimator the standalone polish loop descends with.

Budget consequence, stated up front: a gradient costs 2k+1 = 41 evals and ~12
minutes. The campaign's 160-eval / 480-second candidate budget cannot fit one. This
run therefore raises the per-candidate budget to 400 evals / 2400 s, which is a
DELIBERATE break in comparability with the A/B arms — arm-vs-arm numbers from that
campaign are not commensurable with this run's. What is commensurable is the honest
score of the boundary that comes out, and the resume point is identical (the Plan-1
deep champion, private 0.6408 at feasibility 0.0079).

The wiki is snapshotted before the run so a paired control (STELLAR_FULL_GRAD=0) can
still be run later against a clean memory — the shared-wiki leak that the A/B driver
had to design around applies here too.

  setsid nohup .venv/bin/python experiments/gradrun.py >> runs/gradrun/driver.log 2>&1 &
  .venv/bin/python experiments/gradrun.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "runs" / "gradrun"
SNAP = OUT / "memory-snapshot"
MEMDIR = _ROOT / "memory" / "stellar_p2"
POLISH_STATE = _ROOT / "runs" / "qi_polish" / "state.json"

CHAMPION = "file:runs/plan1-deep-s7-85609652/best.py"

# Per-candidate budget: one gradient is 41 evals / ~12 min, so 400/2400 leaves room
# for ~3 re-linearizations plus a normal search. eval_timeout 180 for the same
# reason the campaign used it — this box stretches 12-27s evals under load.
OVERRIDES = '{"max_evals":400,"cpu_budget":2400.0,"eval_timeout":180.0}'
MARGIN = '{"target":0.002,"slope":0.92}'
NOVELTY = '{"min":0.003,"pen":0.05}'
MEM_MB = "6144"          # pool + a second jax process for the gradient service

USD, CALLS, SECONDS, STALL = 8.0, 400, 172800, 25
GLOBAL_USD = 26.0

# Three runs, not one. The previous campaign's verdict died at n=2 with opposite
# signs, so g1/g2 are independent REPLICATES: same resume point, same budget,
# different seed, and the wiki is restored from the snapshot before each so one
# does not read the other's notes (the shared-wiki leak this harness has). g3 is
# not a replicate — it resumes whichever of the two ended higher and keeps the
# accumulated memory, to see whether gradient gains compound across runs.
STEPS = [
    {"name": "g1", "seed": 29, "resume": CHAMPION, "restore": True},
    {"name": "g2", "seed": 31, "resume": CHAMPION, "restore": True},
    {"name": "g3", "seed": 37, "resume": "@best", "restore": False},
]


def log(*a: object) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]", *a, flush=True)


def polish_running() -> bool:
    """True while the FD polish loop still owns the box's spare CPUs."""
    r = subprocess.run(["pgrep", "-f", "polish_supervise.sh"],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def polish_status() -> str:
    try:
        t = json.loads(POLISH_STATE.read_text())["trajectory"]
        acc = [r for r in t if r.get("accepted")]
        return (f"step {max(r['step'] for r in t)}, {len(acc)} accepted, "
                f"honest {acc[-1]['honest']:.6f}" if acc else "no accepted steps")
    except Exception as e:
        return f"unreadable ({type(e).__name__})"


def exec_run(name: str, seed: int, resume: str) -> int:
    """One run, in a subprocess whose env was set by the parent — task.py reads
    STELLAR_FULL_GRAD at import, so the flag cannot be flipped inside a live
    interpreter (the same one-arm-per-process rule the A/B driver needed)."""
    from core import loop
    from core.config import Config

    cfg = Config(
        task="stellar_p2", seed=seed,
        switches={"feedback": "memory", "gate": "holdout", "search": "staged",
                  "knowledge": "wiki_fs", "roles": "single_strong"},
        budget={"max_usd": USD, "max_calls": CALLS, "max_seconds": SECONDS},
        resume_from=resume,
        stall_stop=STALL,
    )
    summary = loop.run(cfg, run_dir=OUT / name)
    print("RUN_SUMMARY " + json.dumps(
        {k: summary.get(k) for k in ("best_id", "train", "val", "private",
                                     "usd", "calls", "seconds", "stop_reason")},
        default=str), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-run", action="store_true")
    ap.add_argument("--name", default="g1")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--resume", default=CHAMPION)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--now", action="store_true",
                    help="do not wait for the polish loop")
    a = ap.parse_args()

    if a.exec_run:
        return exec_run(a.name, a.seed, a.resume)
    if a.status:
        st = json.loads((OUT / "state.json").read_text()) if (OUT / "state.json").exists() else {}
        print(json.dumps({"polish_running": polish_running(),
                          "polish": polish_status(), "gradrun": st}, indent=2))
        return 0

    OUT.mkdir(parents=True, exist_ok=True)

    while not a.now and polish_running():
        log(f"waiting for the polish loop ({polish_status()})")
        time.sleep(300)
    log(f"polish loop done: {polish_status()}")

    if not SNAP.exists():
        shutil.copytree(MEMDIR, SNAP)
        log(f"wiki snapshotted to {SNAP} for a future paired control")

    # Full golden suite, here and not earlier: several of these tests spawn eval
    # containers and one (test_cvrp_seed_and_validator) is a wall-clock check that
    # fails under CPU contention, so the only honest place to run it is a quiet
    # box — which is exactly this moment, after the polish loop and before
    # anything else is launched. Gating on it is deliberate: nothing is lost by
    # refusing to start a $8 / 20h run on a red suite.
    # The gates below are idempotent: a driver restart (e.g. to pick up a tool
    # fix before the replicates diverge) must not re-spend 8 minutes of suite
    # and 4 minutes of official evals it already passed.
    if (OUT / "tests.ok").exists():
        log("skip test suite (already passed this pipeline)")
    else:
        log("running the full test suite on the now-quiet box...")
        t0 = time.time()
        tests = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                               cwd=str(_ROOT), capture_output=True, text=True,
                               timeout=5400)
        (OUT / "tests.log").write_text(tests.stdout + tests.stderr)
        tail = [ln for ln in tests.stdout.strip().splitlines() if ln.strip()]
        log(f"tests rc={tests.returncode} in {time.time() - t0:.0f}s: "
            + (tail[-1][:200] if tail else "no output"))
        if tests.returncode != 0:
            log(f"TESTS FAILED — not launching. See {OUT}/tests.log")
            return 1
        (OUT / "tests.ok").write_text(tail[-1] if tail else "passed")

    # Does the polish loop's vlf gain survive official fidelity? Two evals, and
    # it needs the box to itself — an official solve wants the full 4 GB, which
    # is why it could not be run beside the polish container. Measured, not
    # gating: it decides whether MORE polishing is worth funding, not this run.
    if (_ROOT / "runs" / "qi_polish" / "transfer.json").exists():
        log("skip transfer check (already measured)")
    else:
        log("measuring vlf -> official transfer of the polish trajectory...")
        tr = subprocess.run([sys.executable, "experiments/transfer_check.py"],
                            cwd=str(_ROOT), capture_output=True, text=True)
        (OUT / "transfer.log").write_text(tr.stdout + tr.stderr)
        log(f"transfer check rc={tr.returncode}: "
            + (tr.stdout.strip().splitlines() or ["no output"])[-1][:300])

    # Gate: a run that discovers the tools are broken burns hours to learn it.
    # The smoke is ~6 solves and exercises the whole path (file shipping, the
    # child process, jax beside the fork pool, a step that actually moves).
    log("running the gradient smoke before committing the run...")
    smoke = subprocess.run([sys.executable, "experiments/_gradsmoke.py"],
                           cwd=str(_ROOT), capture_output=True, text=True)
    (OUT / "smoke.log").write_text(smoke.stdout + smoke.stderr)
    if smoke.returncode != 0:
        log(f"SMOKE FAILED (rc={smoke.returncode}) — not launching. See {OUT}/smoke.log")
        return 1
    log("smoke passed")

    env = dict(os.environ)
    env["STELLAR_FULL_GRAD"] = "1"          # <- the treatment
    env["STELLAR_MARGIN_GRAD"] = "1"        # aspect tools stay on: complementary
    env["STELLAR_TRAIN_OVERRIDES"] = OVERRIDES
    env["STELLAR_MARGIN"] = MARGIN
    env["STELLAR_NOVELTY"] = NOVELTY
    env["STELLAR_MEM_MB"] = MEM_MB
    env["STELLAR_HOT_RESTART"] = "1"
    env["STELLAR_SOFT_FAIL"] = "1"

    state_p = OUT / "state.json"
    st = json.loads(state_p.read_text()) if state_p.exists() else {"done": {}, "usd": 0.0}

    for step in STEPS:
        name = step["name"]
        if name in st["done"]:
            log(f"skip {name} (already done)")
            continue
        if st["usd"] >= GLOBAL_USD:
            log(f"global cap ${GLOBAL_USD} reached (${st['usd']:.2f}) — stopping")
            break

        resume = step["resume"]
        if resume == "@best":
            done = [(r.get("private") or float("-inf"), n)
                    for n, r in st["done"].items() if r.get("best_id")]
            if not done:
                log(f"skip {name}: no completed run to resume from")
                continue
            resume = f"file:runs/gradrun/{max(done)[1]}/best.py"

        if step["restore"] and SNAP.exists():
            # replicate isolation: g2 must not read g1's notes
            shutil.rmtree(MEMDIR, ignore_errors=True)
            shutil.copytree(SNAP, MEMDIR)
            log(f"wiki restored from snapshot for {name}")

        log(f"--- {name}: seed={step['seed']} resume={resume} usd=${USD} "
            f"stall={STALL}")
        t0 = time.time()
        logf = OUT / f"{name}.log"
        with logf.open("a") as fh:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--exec-run", "--name", name,
                                "--seed", str(step["seed"]), "--resume", resume],
                               cwd=str(_ROOT), env=env, stdout=fh,
                               stderr=subprocess.STDOUT)
        summary: dict = {}
        for line in logf.read_text().splitlines()[::-1]:
            if line.startswith("RUN_SUMMARY "):
                summary = json.loads(line[len("RUN_SUMMARY "):])
                break
        rec = {"exit": p.returncode, "wall_s": round(time.time() - t0, 1),
               "seed": step["seed"], "resume": resume, **summary}
        st["done"][name] = rec
        st["usd"] = round(st["usd"] + float(summary.get("usd") or 0.0), 4)
        state_p.write_text(json.dumps(st, indent=2))
        log(f"--- {name} done: " + json.dumps(rec))
        if p.returncode != 0 and not summary:
            log(f"{name} produced no summary (exit {p.returncode}) — stopping the "
                f"campaign rather than compounding a broken state")
            break

    log(f"=== campaign done: {len(st['done'])}/{len(STEPS)} runs, "
        f"${st['usd']:.2f} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
