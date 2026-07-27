"""DAG branch-and-merge campaign driver (stellar_p2, 2026-07-24).

Design (user spec): 6 leaf branches — B1 resumes the s17 incumbent (phanerozoic-basin
margin-aware + momentum polish), B2-B6 are pinned to 5 promising & structurally
different bank seeds. Each branch is developed sequentially under the 25-rule
(stall_stop=25: run until 25 consecutive candidates bring no new combined best),
with GLM writers, an Opus refiner synthesizing every 5-candidate window, an Opus
in-run analyst every 12 candidates (10 GLM + ~2 refiner) that reviews the whole
history, does web research and injects exactly ONE candidate for the next writers
to build on, and wiki reviews every 10 candidates. After each branch a headless Opus session
analyses the branch's runs, cleans the wiki, and writes proposals whose HINT
section steers the next branch. Terminated branches then merge pairwise in
termination order (winner's best resumes, loser's best is injected as Approach B
with a merge directive) under the same 25-rule, until ONE branch remains.

Run:  setsid nohup .venv/bin/python experiments/dag_campaign.py >> runs/dag/driver.log 2>&1 &
State: runs/dag/state.json (resumable — re-running continues where it left off).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "http://localhost:8777/api"
DAG = ROOT / "runs" / "dag"
PROPOSALS = DAG / "proposals"
PINS = ROOT / "experiments" / "branch_seeds"
STATE = DAG / "state.json"

SWITCHES = {"feedback": "memory", "gate": "holdout", "search": "staged",
            "knowledge": "off", "roles": "single_strong"}
# 2026-07-27 (user directive): every role runs on z.ai with the z.ai key — no
# Claude-billed sessions anywhere in the campaign. Refiner/analyst/post-branch
# calls are now on the same bill as the writers, so BRANCH_USD/GLOBAL_USD below
# cover the WHOLE campaign rather than just the writer half of it.
REFINER = "glm-5.2"               # windowed v2 pass every REFINER_EVERY writer cands
REFINER_EVERY = 5
ANALYST = "glm-5.2"               # in-run analysis + single-candidate injection
ANALYST_EVERY = 12                # ~10 writer + 2 refiner candidates per analyst window
POST_BRANCH_MODEL = "glm-5.2"     # branch post-mortem + HINT for the next branch
STALL = 60                        # branch-termination rule (25→40→60 2026-07-25:
# user — let each branch make many iterations, at least 50 candidates, before
# a no-progress stall can terminate it; caps below raised to match so nothing
# cuts a branch short of that)
RUN_USD, RUN_CALLS, RUN_SECONDS = 20.0, 800, 64800  # 18h: MUST stay below the
# driver's RUN_TIMEOUT_S (20h) net so a healthy long run ends on its own budget
# ("budget: max_seconds" => branch CONTINUES from its best) instead of tripping
# the STOP net ("user_stop" => whole campaign HALTS). The 20h net now only
# catches genuinely hung runs. (2026-07-26: raising RUN_SECONDS above the net
# guillotined s103-473410 at 129 cands / private 0.6352 and paused the campaign.)
BRANCH_USD = 30.0                 # per-branch z.ai cap across continuation runs
GLOBAL_USD = 120.0               # whole-campaign z.ai cap (user 2026-07-24:
# novelty > speed — "allow yourself your needed time")
GENERATIONS = 200
POLL_S, RUN_TIMEOUT_S = 45, 20 * 3600

# Campaign 2 leaves (2026-07-27, honest-score pipeline). Campaign 1's verdict:
# its 0.6400 champion bought its whole margin over davidkh's 0.6361 from the
# feasibility tolerance (93% of it vs his 7.5%) and sat cosine 0.999989 from his
# public boundary. Under the margin-discounted fitness that candidate is worth
# honest 0.6320 — BELOW the public bar. So this campaign's target is explicit:
#   honest_score >= 0.6361  AND  bank_dist >= 1e-3  AND  bank_cos clearly < 0.9999.
# N1/N2 restart campaign 1's two strongest evolved OPTIMIZERS under the new
# fitness — the machinery is worth keeping, the tolerance camping is not; they
# re-derive where their score comes from. N3/N4 are fresh basins picked for
# margin headroom (both zero-violation seeds, so nothing to unlearn). N5 is the
# unaided path. Bank facts (official / violation of 0.010):
#  #0 davidkh    0.6361 / 0.00075  — leaderboard #1 boundary itself
#  #3 lhhhhappy  0.6257 / 0.0      — zero violation: maximum margin headroom
#  #4 RisoLiao   0.6236 / 0.0      — zero violation, distinct author basin
#  #6 RisoLiao   0.6071 / 0.0      — earlier, structurally different RisoLiao optimum
#  #8 Elahehkazemi 0.4989 / 0.0    — most recent independent basin
LEAVES = [
    {"name": "N1-honest-champion", "resume": "stellar_p2-s105-26196944", "pin": None,
     "desc": "campaign-1 champion program (c0045, official 0.6400 at feasibility "
             "0.0093 => honest 0.6320) re-run under margin-discounted fitness: "
             "keep the two-stage m-contraction machinery, pay back the tolerance"},
    {"name": "N2-honest-risoliao4", "resume": "stellar_p2-s105-72881323", "pin": None,
     "desc": "campaign-1 B4 lineage (c0088r1, official 0.6398) — second evolved "
             "optimizer, distinct RisoLiao #4 basin, same honest re-derivation"},
    {"name": "N3-lhhhhappy3", "pin": 3,
     "desc": "bank #3 lhhhhappy 0.6257, violation 0.0: the seed whose profile the "
             "honest metric most favours — full 0.008 of unspent margin headroom"},
    {"name": "N4-elaheh8", "pin": 8,
     "desc": "bank #8 Elahehkazemi 0.4989, violation 0.0: the most independent "
             "public basin — low score but the furthest from the davidkh shape"},
    {"name": "N5-nae-independent", "pin": "nae",
     "desc": "independent basin: NAE seeds only, no bank triage — the only fully "
             "own-result path; expect negative shaped scores, terminate fast if stuck"},
]

_PIN_ANCHOR = '        info = sorted(fm.seed_bank_info(), key=lambda e: -e["official_score"])'

POST_BRANCH_PROMPT = """You are the post-branch analysis session of a DAG evolution \
campaign on ConStellaration P2. Branch {name} just terminated ({desc}).

The branch's candidate history is summarised below (host-extracted from the run \
ledgers) together with the memory wiki. Write ONLY a markdown page with exactly \
these sections:
# Branch {name} post-mortem
## Analysis
## HINT

Analysis: what was tried, what won, what was wasted, how the refiner and analyst \
performed. Ground every claim in the candidate rows below — no generic advice.
HINT (max 1200 chars): concrete directives for the NEXT branch's writers — what to \
exploit, what to avoid.

NOTE: this session cannot edit files, so do NOT propose wiki edits as if you were \
making them; anything the wiki must record, say plainly in Analysis so a human or \
the in-run reviewer can fold it in.

# Branch runs
{run_dirs}

# Candidate history
{history}

# Memory wiki
{wiki}"""


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def api(path, body=None, timeout=30):
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"branches": [dict(l, runs=[], status="pending") for l in LEAVES],
            "merge_n": 0, "queue": [], "halt": None}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1))


def make_pin(idx) -> str:
    src = (ROOT / "tasks/stellar_p2/seed/optimizer.py").read_text()
    assert _PIN_ANCHOR in src, "seed template changed: pin anchor line not found"
    if idx == "nae":  # independent-basin branch: force the NAE fallback path
        pinned = src.replace(
            _PIN_ANCHOR,
            "        info = []  # DAG: independent basin — no bank triage")
    else:
        pinned = src.replace(
            _PIN_ANCHOR,
            _PIN_ANCHOR + f'\n        info = [e for e in info if e["i"] == {idx}] or info')
    PINS.mkdir(parents=True, exist_ok=True)
    f = PINS / f"pin{idx}.py"
    f.write_text(pinned + f"\n# DAG campaign: pinned to bank seed #{idx}\n")
    return f"file:experiments/branch_seeds/pin{idx}.py"


def spent(st) -> float:
    return round(sum(r.get("usd") or 0.0 for b in st["branches"] for r in b["runs"]), 2)


def run_end_of(run_id: str) -> dict | None:
    led = ROOT / "runs" / run_id / "ledger.jsonl"
    if not led.exists():
        return None
    for line in reversed(led.read_text().splitlines()[-5:]):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "run_end":
            return ev
    return None


# A working run writes a ledger event at least every candidate (~10-12 min at
# stellar_p2 eval cost). Two hours of silence means the run is gone — the server
# was restarted, the thread died, or it is wedged. Same threshold decides "still
# live, reattach" and "dead, move on", so the two can never both be false.
_LIVE_S = 7200


def _ledger_age(run_id: str) -> float:
    led = ROOT / "runs" / run_id / "ledger.jsonl"
    return time.time() - led.stat().st_mtime if led.exists() else 9e9


def _is_live(run_id: str) -> bool:
    """True if run_id has no run_end and its ledger is still being written."""
    return run_end_of(run_id) is None and _ledger_age(run_id) < _LIVE_S


def _has_candidate(run_id: str) -> bool:
    """resume_from needs a candidate to resume from; a run that produced none
    would 422 the launch and crash the driver on its retry loop."""
    led = ROOT / "runs" / run_id / "ledger.jsonl"
    if not led.exists():
        return False
    return '"type": "candidate"' in led.read_text()


def wait_for(run_id: str) -> dict:
    t0 = time.time()
    stopped = False
    while True:
        ev = run_end_of(run_id)
        if ev is not None:
            return ev
        if _ledger_age(run_id) > _LIVE_S:
            # Do not sit on a corpse for 20h. Note the stop_reason deliberately
            # does NOT start with "stall": that word terminates the branch, and a
            # dead run is not evidence the branch is out of ideas.
            log(f"DEAD {run_id}: no ledger activity for {_LIVE_S / 3600:.0f}h")
            (ROOT / "runs" / run_id / "STOP").touch()
            return {"stop_reason": f"dead: no ledger activity for {_LIVE_S / 3600:.0f}h",
                    "usd": 0.0, "seconds": round(time.time() - t0)}
        if time.time() - t0 > RUN_TIMEOUT_S and not stopped:
            log(f"TIMEOUT {run_id}: touching STOP")
            (ROOT / "runs" / run_id / "STOP").touch()
            stopped = True
        if time.time() - t0 > RUN_TIMEOUT_S + 3600:
            raise RuntimeError(f"run {run_id} never ended")
        time.sleep(POLL_S)


def launch(cfg: dict, hint: str | None) -> str:
    for attempt in range(3):
        try:
            run_id = api("/runs", cfg)["run_id"]
            break
        except Exception as e:
            log(f"launch failed ({e}), retry {attempt + 1}/3")
            time.sleep(60)
    else:
        raise RuntimeError("API unreachable: cannot launch run")
    if hint:
        (ROOT / "runs" / run_id / "HINT").write_text(hint[:1900])
    return run_id


def best_run(branch: dict) -> dict:
    def key(r):
        return (r.get("private") if r.get("private") is not None else -9e9,
                r.get("val") if r.get("val") is not None else -9e9,
                r.get("train") or -9e9)
    return max(branch["runs"], key=key)


def branch_hint(st, branch) -> str:
    parts = [f"DAG campaign branch {branch['name']}: {branch['desc']}. "
             f"Terminate rule: {STALL} candidates without a new best ends the branch — "
             "make every candidate count; surgical, distinct, evidence-grounded ideas.",
             "THE BAR (the campaign's whole point, tightened 2026-07-27): "
             "honest_score >= 0.6361 AND max-coeff distance >= 1e-3 from EVERY bank "
             "seed AND a bank_cos clearly below 0.9999. honest_score = p2_score - "
             "0.92*max(0, feasibility - 0.002) is what train/val fitness now IS: the "
             "previous campaign's 0.6400 came from spending 93% of the 1% feasibility "
             "tolerance and is only honest 0.6320 — below the public bar it appeared "
             "to beat, and cosine 0.999989 from davidkh's public boundary. Do not "
             "repeat that. Buy score with structure at feasibility <= 0.002; treat "
             "aspect-ratio tolerance as already spent. Near-copies pay up to 0.05 "
             "novelty penalty at train/val (metrics: bank_dist, bank_cos, "
             "novelty_penalty) and are refused at export."]
    if branch.get("merge_from_run"):
        parts.append("This is a MERGE branch: combine Approach A (incumbent) with "
                     "Approach B shown in the prompt — integrate mechanisms, don't pick one.")
    prev = [b for b in st["branches"] if b["status"] == "done" and b.get("hint_out")]
    if prev:
        parts.append(f"Post-mortem of previous branch {prev[-1]['name']}:\n"
                     + prev[-1]["hint_out"])
    return "\n\n".join(parts)


def run_branch(st, branch):
    branch["status"] = "running"
    save_state(st)
    seed_n = 100 + sum(len(b["runs"]) for b in st["branches"])
    while True:
        if spent(st) >= GLOBAL_USD:
            st["halt"] = f"global usd cap {GLOBAL_USD} reached"
            branch["status"] = "capped"
            return
        busd = sum(r.get("usd") or 0.0 for r in branch["runs"])
        # Resume from the most recent run that actually produced a candidate: a
        # run that died before its first one has nothing to resume FROM, and
        # pointing resume_from at it 422s the launch three times and crashes the
        # driver. Fall back through the branch's history to its original seed.
        resume = branch["resume"]
        for r in reversed(branch["runs"]):
            if _has_candidate(r["run_id"]):
                resume = r["run_id"]
                break
        cfg = {"task": "stellar_p2", "seed": seed_n + len(branch["runs"]),
               "switches": SWITCHES,
               "budget": {"max_usd": min(RUN_USD, BRANCH_USD - busd),
                          "max_calls": RUN_CALLS, "max_seconds": RUN_SECONDS},
               "objective": "quality", "generations": GENERATIONS,
               "stall_stop": STALL, "review_every": 10,
               "analyst": ANALYST, "analyst_every": ANALYST_EVERY,
               "analyst_web": True, "analyst_inject": 1,
               "refiner": REFINER, "refiner_every": REFINER_EVERY,
               "resume_from": resume}
        if branch.get("merge_from_run"):
            cfg["merge_from"] = branch["merge_from_run"]
        # Attach to an in-flight run instead of starting a second one. The driver
        # may be restarted (code change, crash) while the server thread it
        # launched is still evaluating; relaunching then puts two stellar runs on
        # the same 4 vCPUs, which is the documented way to make every eval time
        # out. main()'s recovery leaves the live run's stop_reason unset for us.
        last = branch["runs"][-1] if branch["runs"] else None
        if last and not last.get("stop_reason") and _is_live(last["run_id"]):
            run_id = last["run_id"]
            log(f"{branch['name']}: attaching to in-flight {run_id}")
        else:
            run_id = launch(cfg, branch_hint(st, branch))
            log(f"{branch['name']}: launched {run_id} (resume={resume})")
            branch["runs"].append({"run_id": run_id})
            save_state(st)
        end = wait_for(run_id)
        branch["runs"][-1].update(
            stop_reason=end.get("stop_reason"), best_id=end.get("best_id"),
            train=end.get("train"), val=end.get("val"), private=end.get("private"),
            usd=end.get("usd"), calls=end.get("calls"), seconds=end.get("seconds"))
        save_state(st)
        log(f"{branch['name']}: {run_id} ended — {end.get('stop_reason')!r} "
            f"best {end.get('best_id')} val {end.get('val')} private {end.get('private')} "
            f"${end.get('usd')}  (campaign ${spent(st)})")
        sr = str(end.get("stop_reason") or "")
        if sr.startswith("stall"):
            branch["status"] = "done"
            return
        if sr == "user_stop":
            branch["status"] = "user_stop"
            st["halt"] = "user stopped a run — campaign paused"
            return
        # Provider out of money: every relaunch fails the same way in seconds, so
        # the driver would burn the branch's run list spinning. Campaign 1 did
        # exactly this on 2026-07-27 until a human noticed. Halt instead.
        if "1113" in sr or "nsufficient balance" in sr:
            branch["status"] = "pending"
            st["halt"] = ("z.ai balance exhausted (429/1113) — recharge, then "
                          "relaunch the driver; it reattaches and continues")
            return
        # Any other run that dies fast and free is a broken environment, not
        # progress: two in a row means stop asking.
        if (end.get("seconds") or 0) < 300 and (end.get("usd") or 0.0) < 0.05:
            branch["dud_streak"] = branch.get("dud_streak", 0) + 1
            if branch["dud_streak"] >= 2:
                branch["status"] = "pending"
                st["halt"] = (f"{branch['name']}: two runs ended in <5min for <$0.05 "
                              f"(last: {sr[:120]!r}) — environment problem, not search")
                return
        else:
            branch["dud_streak"] = 0
        if sum(r.get("usd") or 0.0 for r in branch["runs"]) >= BRANCH_USD - 0.05:
            branch["status"] = "done"  # budget-terminated counts as terminated
            branch["capped"] = True
            return
        # budget/generations/llm_error: continue the branch from this run


def _branch_history(branch) -> str:
    """One line per candidate across the branch's runs — the post-mortem's evidence."""
    lines = []
    for r in branch["runs"]:
        p = ROOT / "runs" / r["run_id"] / "ledger.jsonl"
        if not p.exists():
            continue
        lines.append(f"--- {r['run_id']} ---")
        for raw in p.read_text().splitlines():
            try:
                e = json.loads(raw)
            except ValueError:
                continue
            if e.get("type") != "candidate":
                continue
            m, s = e.get("meta") or {}, e.get("scores") or {}
            role = "refiner" if m.get("refiner") else ("analyst" if "i" in
                                                       str(e.get("id"))[5:] else "writer")
            met = {k: v for k, v in (m.get("metrics") or {}).items()
                   if k in ("p2_score", "feasibility", "bank_dist")}
            lines.append(
                f"{e.get('id')} [{role}] {'acc' if e.get('accepted') else 'rej'} "
                f"train={s.get('train')} val={s.get('val')} {met} "
                f"err={(str(m.get('error') or ''))[:60]} idea={(m.get('idea') or '')[:160]}")
    return "\n".join(lines)[:60000] or "(no candidate events)"


def _wiki_text() -> str:
    mem = ROOT / "memory" / "stellar_p2"
    idx = mem / "index.md"
    if not idx.exists():
        return "(no wiki)"
    parts = [f"--- index.md ---\n{idx.read_text()[:5000]}"]
    for p in sorted(mem.glob("*/**/*.md")) + sorted(mem.glob("*/*.md")):
        parts.append(f"--- {p.relative_to(mem)} ---\n{p.read_text()[:2000]}")
    return "\n\n".join(dict.fromkeys(parts))[:40000]


def post_branch(st, branch):
    """z.ai post-mortem (2026-07-27: was a headless Claude session with file-edit
    tools). The model no longer edits the wiki — that capability cost more than it
    returned (it repeatedly deleted pages without merging them) and Claude sessions
    are off the table; it writes the analysis + HINT, the host writes the file."""
    PROPOSALS.mkdir(parents=True, exist_ok=True)
    proposal = PROPOSALS / f"{branch['name']}.md"
    run_dirs = ", ".join(f"runs/{r['run_id']}" for r in branch["runs"])
    prompt = POST_BRANCH_PROMPT.format(
        name=branch["name"], desc=branch["desc"], run_dirs=run_dirs,
        history=_branch_history(branch), wiki=_wiki_text())
    log(f"{branch['name']}: post-branch z.ai session starting ({POST_BRANCH_MODEL})")
    try:
        sys.path.insert(0, str(ROOT))
        from core.ledger import BudgetGuard, Ledger
        from core.llm import LLM
        llm = LLM(Ledger(DAG), BudgetGuard(max_usd=2.0, max_calls=10,
                                           max_seconds=1800))
        text = llm.chat(POST_BRANCH_MODEL, [{"role": "user", "content": prompt}],
                        0.4, role="post_branch", max_tokens=8192)
        if text.strip():
            proposal.write_text(text.strip() + "\n")
    except Exception as e:
        log(f"{branch['name']}: post-branch session failed: {e}")
    if proposal.exists():
        text = proposal.read_text()
        hint = text.split("## HINT", 1)[-1].strip() if "## HINT" in text else ""
        branch["hint_out"] = hint[:1500]
        log(f"{branch['name']}: proposals written ({len(text)} chars)")
    else:
        log(f"{branch['name']}: no proposals file produced")
    save_state(st)


def main():
    DAG.mkdir(parents=True, exist_ok=True)
    st = load_state()
    for b in st["branches"]:
        if b.get("pin") is not None and not b.get("resume"):
            b["resume"] = make_pin(b["pin"])
    save_state(st)
    # recover orphaned runs (driver or server restarted mid-run)
    for b in st["branches"]:
        if b["status"] == "running":
            last = b["runs"][-1] if b["runs"] else None
            if last and not last.get("stop_reason"):
                ev = run_end_of(last["run_id"])
                if ev:
                    last.update(stop_reason=ev.get("stop_reason"),
                                best_id=ev.get("best_id"), train=ev.get("train"),
                                val=ev.get("val"), private=ev.get("private"),
                                usd=ev.get("usd"))
                elif _is_live(last["run_id"]):
                    # still evaluating in the server: leave stop_reason unset so
                    # run_branch reattaches rather than starting a rival run
                    log(f"run {last['run_id']} is still live: will reattach")
                else:
                    led = ROOT / "runs" / last["run_id"] / "ledger.jsonl"
                    if led.exists() and time.time() - led.stat().st_mtime > 7200:
                        log(f"orphaned run {last['run_id']}: will resume branch from it")
                        last.update(stop_reason="orphaned", usd=last.get("usd") or 0.0)
            b["status"] = "pending" if not str(
                (b["runs"][-1].get("stop_reason") if b["runs"] else "") or "").startswith(
                "stall") else "done"
    save_state(st)

    # phase 1: leaves, sequential
    # leaves = every branch that is not a merge (was: pin is not None or name
    # startswith "B" — which silently skipped campaign-2's resume-based leaves,
    # whose pin is None and whose names start with "N")
    for b in [x for x in st["branches"] if not x.get("merge_from_run")]:
        if b["status"] in ("done", "capped", "user_stop"):
            continue
        run_branch(st, b)
        save_state(st)
        if st["halt"]:
            log(f"HALT: {st['halt']}")
            return
        post_branch(st, b)

    # phase 2: pairwise merge tournament in termination order, until one branch left
    if not st["queue"]:
        st["queue"] = [b["name"] for b in st["branches"] if b["status"] == "done"]
    by_name = {b["name"]: b for b in st["branches"]}
    while len(st["queue"]) > 1:
        a_name, b_name = st["queue"][0], st["queue"][1]
        a, b = by_name[a_name], by_name[b_name]
        ra, rb = best_run(a), best_run(b)
        ka = (ra.get("private") or -9, ra.get("val") or -9)
        kb = (rb.get("private") or -9, rb.get("val") or -9)
        win, lose, rw, rl = (a, b, ra, rb) if ka >= kb else (b, a, rb, ra)
        st["merge_n"] += 1
        m = {"name": f"M{st['merge_n']}-{win['name']}x{lose['name']}",
             "desc": f"merge: {win['name']} (best {rw.get('val')}) resumes; "
                     f"{lose['name']} (best {rl.get('val')}) injected as Approach B",
             "resume": rw["run_id"], "merge_from_run": rl["run_id"],
             "pin": None, "runs": [], "status": "pending"}
        st["branches"].append(m)
        by_name[m["name"]] = m
        st["queue"] = st["queue"][2:]
        save_state(st)
        log(f"MERGE {m['name']}: {m['desc']}")
        run_branch(st, m)
        save_state(st)
        if st["halt"]:
            log(f"HALT: {st['halt']}")
            return
        post_branch(st, m)
        st["queue"].append(m["name"])
        save_state(st)

    final = by_name[st["queue"][0]] if st["queue"] else None
    if final:
        r = best_run(final)
        (DAG / "FINAL.md").write_text(
            f"# DAG campaign final branch: {final['name']}\n\n"
            f"Best run {r['run_id']} best_id {r.get('best_id')} "
            f"train {r.get('train')} val {r.get('val')} PRIVATE {r.get('private')}\n"
            f"Campaign z.ai spend: ${spent(st)}\n"
            f"Export: python experiments/submit_export.py runs/{r['run_id']}\n")
        log(f"CAMPAIGN DONE — final branch {final['name']}, "
            f"private {r.get('private')}, total ${spent(st)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"DRIVER CRASH: {e!r}")
        sys.exit(1)
