"""Gradient-feedback A/B campaign: does the optimizer do better when handed the
exact boundary-margin gradient tools (fm.aspect / fm.margin_grad / fm.margin_step)?

Design notes — why it looks like this:

* PAIRED A/B, not a single treated arm. Each pair runs the SAME resume point and
  the SAME seed twice: once with STELLAR_MARGIN_GRAD=1 (tools available and
  documented) and once with =0 (methods refuse AND the doc section is stripped,
  so the control writer is never told the tools exist). Arm order alternates
  between pairs so a drifting box load cannot masquerade as a treatment effect.
* ONE ARM PER PROCESS. task.py reads STELLAR_MARGIN_GRAD at import
  (tasks/stellar_p2/task.py:66), so a second arm in the same interpreter would
  silently inherit the first arm's setting. Each arm is therefore a subprocess
  of this driver (`--exec-arm`), never an in-process second call.
* MEMORY IS FROZEN across the A/B. The memory axis keeps ONE shared wiki per
  task at memory/stellar_p2/ that every run reads and extends — so without care
  the treated arm's notes ("use fm.margin_grad") leak into the control's
  prompts, and the archive/novelty state diverges between arms. The driver
  snapshots memory/stellar_p2 once at campaign start and restores that exact
  snapshot before every A/B arm. Each arm's resulting memory is preserved in
  <run_dir>/memory-after/ first, so nothing is lost — only quarantined.
  The final unpaired PUSH step (leaderboard, not science) runs with accumulated
  memory: it is a best-effort score run, not a controlled comparison.
* PLAN-1 FEATURES ON IN BOTH ARMS (warm evals + graded soft-fail). They are the
  new baseline; holding them fixed keeps the gradient tool the only variable.
  This also means these runs are NOT comparable to the pre-Plan-1 ledger runs
  (s201-s207): that comparison moves two variables at once.
* HALTS ON PROVIDER DEATH, not on a dollar cap: z.ai balance exhaustion
  (429 / error 1113 in the stop reason) or two consecutive runs that ended in
  under 5 minutes for under $0.05. --global-usd is only a runaway backstop.
* RESTARTABLE: runs/grad_campaign/state.json records completed steps; re-running
  the driver continues where it stopped.

Usage:
  .venv/bin/python experiments/grad_campaign.py                # run the campaign
  .venv/bin/python experiments/grad_campaign.py --status       # print progress
  .venv/bin/python experiments/grad_campaign.py --dry-run      # show the plan
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

CAMP = _ROOT / "runs" / "grad_campaign"
STATE = CAMP / "state.json"
SNAP = CAMP / "memory-snapshot"
MEMDIR = _ROOT / "memory" / "stellar_p2"

# The gradient-free champion of the Plan-1 deep runs (c0058, private 0.6408 at
# feasibility 0.0079 — i.e. it spends 79% of the 1e-2 tolerance; the honest
# margin question is exactly what the gradient tools are supposed to move).
CHAMPION = "file:runs/plan1-deep-s7-85609652/best.py"

# Deep-run settings = the CANONICAL campaign config from serve.sh, not the
# Plan-1 pipeline's. Three deltas, all deliberate:
#   eval_timeout 180 (not the 60s default): on this contended box 12-27s
#     bank-seed evals have been pushed past 60s and silently -inf'd (runs s105,
#     s103-*). Swap is currently full, so 60s would poison both arms.
#   STELLAR_MARGIN: honest margin-discounted train/val fitness. This is the
#     default in task.py too, but it is stated explicitly because the whole
#     point of the gradient tool is placing the margin, and a tool that lands
#     the aspect wall to 11 digits is a better CAMPING instrument than the
#     ladder was if the discount is not on.
#   STELLAR_NOVELTY 3e-3: campaign-2 ramp (the 1e-3 default switches off exactly
#     where near-copies of the public #1 boundary live).
# Consequence: plan1-deep-s5/s7 (60s timeout, 1e-3 ramp) are an INDICATIVE
# secondary reference only. The controlled comparison is arm-vs-arm inside this
# campaign, where every one of these settings is identical.
OVERRIDES = '{"max_evals":160,"cpu_budget":480.0,"eval_timeout":180.0}'
MARGIN = '{"target":0.002,"slope":0.92}'
NOVELTY = '{"min":0.003,"pen":0.05}'

STEPS = [
    # (name, arm, seed, resume) — pairs share seed+resume; arm order alternates.
    ("p1-on", "on", 11, CHAMPION),
    ("p1-off", "off", 11, CHAMPION),
    ("p2-off", "off", 13, CHAMPION),
    ("p2-on", "on", 13, CHAMPION),
    # Unpaired best-effort push: accumulated memory, resumes the best program
    # the campaign produced so far (resolved at launch time).
    ("push", "on", 17, "@best"),
]


def log(*a: object) -> None:
    msg = f"[{time.strftime('%F %T')}] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    CAMP.mkdir(parents=True, exist_ok=True)
    with (CAMP / "campaign.log").open("a") as f:
        f.write(msg + "\n")


# --------------------------------------------------------------------------
# arm execution (subprocess side)
# --------------------------------------------------------------------------

def exec_arm(name: str, seed: int, resume: str, usd: float, calls: int,
             seconds: int) -> int:
    """Run ONE arm in this process. The parent has already set the env."""
    from core.config import Config
    from core import loop

    cfg = Config(
        task="stellar_p2", seed=seed,
        switches={"feedback": "memory", "gate": "holdout", "search": "staged",
                  "knowledge": "wiki_fs", "roles": "single_strong"},
        budget={"max_usd": usd, "max_calls": calls, "max_seconds": seconds},
        resume_from=resume,
        stall_stop=25,
    )
    run_dir = CAMP / name
    summary = loop.run(cfg, run_dir=run_dir)
    print("RUN_SUMMARY " + json.dumps(
        {k: summary.get(k) for k in ("best_id", "train", "val", "private",
                                     "usd", "calls", "seconds", "stop_reason")},
        default=str), flush=True)
    return 0


# --------------------------------------------------------------------------
# memory freeze
# --------------------------------------------------------------------------

def snapshot_memory() -> None:
    if SNAP.exists():
        return
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(MEMDIR, SNAP)
    log(f"memory snapshot taken: {SNAP} "
        f"({sum(1 for _ in SNAP.rglob('*') if _.is_file())} files)")


def preserve_and_restore(run_dir: Path) -> None:
    """Keep this arm's memory output next to its run, then reset to the snapshot
    so the next arm starts from byte-identical state."""
    if MEMDIR.exists():
        dst = run_dir / "memory-after"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(MEMDIR, dst)
        shutil.rmtree(MEMDIR)
    shutil.copytree(SNAP, MEMDIR)
    log(f"memory restored from snapshot (arm output kept in {run_dir}/memory-after)")


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"done": {}, "halt": None, "usd": 0.0, "short_streak": 0}


def save_state(st: dict) -> None:
    CAMP.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2, default=str))


def best_so_far(st: dict) -> str | None:
    """Best program across completed campaign steps, by private score."""
    ranked = sorted(
        ((v.get("private"), k) for k, v in st["done"].items()
         if isinstance(v.get("private"), (int, float))
         and (CAMP / k / "best.py").exists()),
        reverse=True)
    return f"file:runs/grad_campaign/{ranked[0][1]}/best.py" if ranked else None


def run_step(st: dict, name: str, arm: str, seed: int, resume: str,
             usd: float, calls: int, seconds: int) -> None:
    run_dir = CAMP / name
    env = dict(os.environ)
    env["STELLAR_MARGIN_GRAD"] = "1" if arm == "on" else "0"
    env["STELLAR_TRAIN_OVERRIDES"] = OVERRIDES
    env["STELLAR_MARGIN"] = MARGIN
    env["STELLAR_NOVELTY"] = NOVELTY
    # Plan-1 features held ON in both arms: the gradient tool is the ONLY variable.
    env["STELLAR_HOT_RESTART"] = "1"
    env["STELLAR_SOFT_FAIL"] = "1"

    log(f"--- step {name}: arm={arm} seed={seed} resume={resume} usd=${usd}")
    t0 = time.time()
    logf = CAMP / f"{name}.log"
    with logf.open("a") as fh:
        p = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--exec-arm",
             "--name", name, "--arm", arm, "--seed", str(seed),
             "--resume", resume, "--usd", str(usd), "--calls", str(calls),
             "--seconds", str(seconds)],
            cwd=str(_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    wall = time.time() - t0

    summary: dict = {}
    for line in logf.read_text().splitlines()[::-1]:
        if line.startswith("RUN_SUMMARY "):
            summary = json.loads(line[len("RUN_SUMMARY "):])
            break
    rec = {"arm": arm, "seed": seed, "resume": resume, "exit": p.returncode,
           "wall_s": round(wall, 1), **summary}
    st["done"][name] = rec
    st["usd"] = round(st.get("usd", 0.0) + float(summary.get("usd") or 0.0), 4)
    log(f"--- step {name} done: exit={p.returncode} wall={wall/60:.1f}min "
        f"train={summary.get('train')} private={summary.get('private')} "
        f"usd={summary.get('usd')} stop={summary.get('stop_reason')!r}")

    # ---- halt conditions -------------------------------------------------
    sr = str(summary.get("stop_reason") or "")
    if "1113" in sr or "nsufficient balance" in sr or "429" in sr:
        st["halt"] = ("z.ai balance exhausted (429/1113) — recharge the key, "
                      "then re-run this driver to continue")
    elif p.returncode != 0 and not summary:
        st["short_streak"] = st.get("short_streak", 0) + 1
        if st["short_streak"] >= 2:
            st["halt"] = "two consecutive arms crashed without producing a run summary"
    elif (summary.get("seconds") or 0) < 300 and (summary.get("usd") or 0) < 0.05:
        st["short_streak"] = st.get("short_streak", 0) + 1
        if st["short_streak"] >= 2:
            st["halt"] = ("two consecutive runs ended in <5min for <$0.05 — "
                          "the provider is almost certainly dead")
    else:
        st["short_streak"] = 0

    if name != "push":
        preserve_and_restore(run_dir)
    save_state(st)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-arm", action="store_true",
                    help="internal: run one arm in this process")
    ap.add_argument("--name")
    ap.add_argument("--arm", choices=["on", "off"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--usd", type=float, default=8.0)
    ap.add_argument("--calls", type=int, default=400)
    ap.add_argument("--seconds", type=int, default=12 * 3600)
    ap.add_argument("--global-usd", type=float, default=80.0,
                    help="runaway backstop only; the intended terminator is "
                         "provider-balance exhaustion")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="comma-separated step names to run")
    a = ap.parse_args()

    if a.exec_arm:
        return exec_arm(a.name, a.seed, a.resume, a.usd, a.calls, a.seconds)

    st = load_state()

    if a.status:
        print(json.dumps(st, indent=2, default=str))
        return 0

    steps = [s for s in STEPS if not a.only or s[0] in a.only.split(",")]
    if a.dry_run:
        for name, arm, seed, resume in steps:
            mark = "DONE" if name in st["done"] else "todo"
            print(f"{mark:5s} {name:8s} arm={arm:3s} seed={seed:3d} resume={resume}")
        print(f"halt={st.get('halt')!r} usd={st.get('usd')}")
        return 0

    if st.get("halt"):
        log(f"campaign is halted: {st['halt']}")
        log("clear state.json's \"halt\" field to resume")
        return 1

    snapshot_memory()
    log(f"=== grad A/B campaign start (pid {os.getpid()}) — "
        f"{len([s for s in steps if s[0] not in st['done']])} steps to go ===")

    for name, arm, seed, resume in steps:
        if name in st["done"]:
            log(f"skip {name} (already done)")
            continue
        if st.get("usd", 0.0) >= a.global_usd:
            st["halt"] = f"global usd backstop {a.global_usd} reached"
            save_state(st)
            break
        if resume == "@best":
            resume = best_so_far(st) or CHAMPION
        run_step(st, name, arm, seed, resume, a.usd, a.calls, a.seconds)
        if st.get("halt"):
            break

    log(f"=== campaign stop: halt={st.get('halt')!r} spent=${st.get('usd')} ===")
    # paired comparison, printed for convenience
    for pair in ("p1", "p2"):
        on, off = st["done"].get(f"{pair}-on"), st["done"].get(f"{pair}-off")
        if on and off:
            log(f"{pair}: ON train={on.get('train')} private={on.get('private')} | "
                f"OFF train={off.get('train')} private={off.get('private')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
