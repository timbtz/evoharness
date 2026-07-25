"""The evolution loop: generate → evaluate → gate → insert → repeat.

Composes the 5 axis objects from the config and never special-cases a switch.
A run is a pure function of (config, task, seed) modulo LLM nondeterminism —
which is why every event lands in the ledger.
"""
from __future__ import annotations

import difflib
import importlib
import random
import re
import time
from pathlib import Path

from axes.feedback import Reflections, ScoreOnly
from axes.gate import HoldoutGate, PublicOnly
from axes.knowledge import Off, WikiFS
from axes.memory import MemoryWiki
from axes.research import Researcher
from axes.roles import SingleStrong, SplitRoles
from axes.search import Greedy11, Islands, Staged
from core.candidate import Candidate, Pool, PromptSection
from core.config import Config
from core.ledger import BudgetExceeded, BudgetGuard, Ledger
from core.llm import LLM, parse_code, parse_idea, parse_prediction, parse_reasoning

_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_EVERY = 5  # generations between public-split reports
COMPILE_REPAIRS = 2  # mandatory baseline, not an axis: a non-compiling candidate is
# repaired in-conversation (writer sees the compiler's stderr) before the gen is spent


def _compile_error(err: str) -> bool:
    """C toolchain failures only: cvrp's compile_c message, or raw gcc diagnostics
    (matmul_c). Runtime/validation errors stay ordinary evolutionary feedback."""
    return "compile failed" in err or (".c:" in err and "error:" in err)

SYSTEM = (
    "You are an expert algorithm designer evolving a function (language given by the task). "
    "Start your response with one line `Idea: <one-sentence summary of the change>`, then "
    "one line `Prediction: <which instances should improve, roughly how much, and the "
    "mechanism — a falsifiable claim>`, then brief reasoning, then respond with exactly "
    "one fenced code block containing the complete improved function and anything it "
    "needs (imports/includes/helpers)."
)

_AXES = {
    "feedback": {"score_only": ScoreOnly, "reflections": Reflections, "memory": MemoryWiki},
    "gate": {"public_only": PublicOnly, "holdout": HoldoutGate},
    "search": {"greedy": Greedy11, "islands": Islands, "staged": Staged},
    "knowledge": {"off": Off, "wiki_fs": WikiFS, "web": Researcher},
    "roles": {"single_strong": SingleStrong, "split_roles": SplitRoles},
}


def load_task(name: str):
    return importlib.import_module(f"tasks.{name}.task").TASK


def _combined(scores: dict) -> float:
    """Selection metric for what a run RETURNS (run-end best, resume seed): mean of
    train and val when val was scored — val is the only large-n signal, so train-only
    selection is blind to exactly what the private split measures. Parent selection
    inside the loop stays train-only (val must stay a holdout for the gate)."""
    t = scores.get("train", float("-inf"))
    v = scores.get("val")
    return (t + v) / 2 if v is not None else t


def resume_code(run_id: str, task_name: str) -> str:
    """Best accepted candidate from a prior run's ledger (fallback: its best.py).
    "run_id:best30" resumes that run's saved 30s-budget champion file instead.
    "file:<path>" (relative to repo root) resumes from a code file directly —
    used by the DAG campaign for pinned seed variants.
    Raises ValueError so the API can reject bad requests before a run starts."""
    if run_id.startswith("file:"):
        p = Path(run_id[5:])
        f = p if p.is_absolute() else _ROOT / p
        if not f.is_file():
            raise ValueError(f"resume_from: no such file {run_id[5:]!r}")
        return f.read_text()
    run_id, _, artifact = run_id.partition(":")
    d = _ROOT / "runs" / Path(run_id).name
    if not (d / "ledger.jsonl").exists():
        raise ValueError(f"resume_from: no ledger for run {run_id!r}")
    best, score = "", float("-inf")
    for ev in Ledger(d).read():
        if ev["type"] == "run_start" and ev["config"].get("task") != task_name:
            raise ValueError(f"resume_from: {run_id!r} is task "
                             f"{ev['config'].get('task')!r}, not {task_name!r}")
        if not artifact and ev["type"] == "candidate" and ev.get("accepted") and ev.get("code") \
                and _combined(ev.get("scores", {})) >= score:
            best, score = ev["code"], _combined(ev["scores"])
    if artifact:
        f = d / f"{Path(artifact).name}.py"
        if not f.exists():
            raise ValueError(f"resume_from: no {artifact}.py in run {run_id!r}")
        return f.read_text()
    if not best and (d / "best.py").exists():
        best = (d / "best.py").read_text()
    if not best:
        raise ValueError(f"resume_from: {run_id!r} has no usable candidate")
    return best


def make_key(objective: str):
    """The scalar the gate and parent-ranking optimize (§6). time_capped == quality
    (the budget guard enforces max_seconds for every objective)."""
    def key(cand: Candidate, pool: Pool, split: str = "train") -> float:
        s = cand.score(split)
        if objective == "quality_per_dollar" and pool.all:
            gain = s - pool.all[0].score(split)
            return gain / max(cand.cost.get("usd") or 0.0, 0.01)
        return s
    return key


def assemble(cfg: Config, task, llm) -> dict:
    key = make_key(cfg.objective)
    sw = cfg.switches
    roles = _AXES["roles"][sw["roles"]](cfg.models)
    return {
        "roles": roles,
        "feedback": _AXES["feedback"][sw["feedback"]](task, llm, roles, cfg.temperatures),
        "gate": _AXES["gate"][sw["gate"]](task, key),
        "search": _AXES["search"][sw["search"]](key, cfg.seed),
        "knowledge": _AXES["knowledge"][sw["knowledge"]](task.wiki_dir, cfg.wiki_mode),
        "key": key,
    }


def _evaluate_into(task, cand: Candidate, split: str = "train") -> None:
    res = task.evaluate(cand.code, split)
    cand.scores[split] = res.score
    cand.meta.update(metrics=res.metrics, seconds=round(res.seconds, 2))
    if res.error:
        cand.meta["error"] = res.error


def _probe_extra_splits(task, cand: Candidate) -> None:
    """Pareto probes (e.g. cvrp val30): scored only for ACCEPTED candidates, so the
    per-candidate cost stays at the cheap splits while the pool still learns which
    survivors generalize across budgets."""
    for split in getattr(task, "extra_splits", ()):
        res = task.evaluate(cand.code, split)
        cand.scores[split] = res.score
        cand.meta[f"metrics_{split}"] = res.metrics


def _report_public(task, ledger: Ledger, best: Candidate, gen: int, guard: BudgetGuard) -> None:
    res = task.evaluate(best.code, "public")
    best.scores["public"] = res.score
    ledger.append({
        "type": "public", "gen": gen, "id": best.id, "score": res.score,
        "metrics": res.metrics, "usd": round(guard.usd, 4),
        "render": task.render(best.code, res),
    })


def _private_median(task, code: str):
    """Run-end truth. Noisy (anytime) tasks: median of 3 — single evals spread
    ~0.1-0.18 gap-pct pts. Deterministic tasks (task.noise empty): one eval is
    the truth; extra ones only pay if the first fails transiently (a sandbox
    hiccup must not report -inf for a working candidate)."""
    n = 3 if getattr(task, "noise", None) else 1
    evals = [task.evaluate(code, "private") for _ in range(n)]
    while not any(r.score > float("-inf") for r in evals) and len(evals) < 3:
        evals.append(task.evaluate(code, "private"))
    ok = [r for r in evals if r.score > float("-inf")] or evals
    return evals, sorted(ok, key=lambda r: r.score)[len(ok) // 2]


def run(cfg: Config, run_dir: str | Path | None = None, llm_factory=LLM) -> dict:
    task = load_task(cfg.task)
    run_dir = Path(run_dir or _ROOT / "runs" / f"{cfg.task}-{cfg.seed}-{int(time.time())}")
    ledger = Ledger(run_dir)
    guard = BudgetGuard(**cfg.budget)
    llm = llm_factory(ledger, guard)
    rng = random.Random(cfg.seed)
    ax = assemble(cfg, task, llm)
    refiner = None
    if cfg.refiner:
        from axes.refiner import ClaudeRefiner
        refiner = ClaudeRefiner(model=cfg.refiner)
        refiner.bind(run_dir, ledger)
    analyst = None
    if cfg.analyst:
        from axes.analyst import ClaudeAnalyst
        analyst = ClaudeAnalyst(model=cfg.analyst, every=cfg.analyst_every,
                                web=cfg.analyst_web, inject=cfg.analyst_inject)
        analyst.bind(run_dir, ledger, task)
    if cfg.review_every:
        ax["feedback"].REVIEW_EVERY = cfg.review_every  # instance attr shadows class
    merge_section = None
    if cfg.merge_from:
        merge_src = resume_code(cfg.merge_from, cfg.task)
        merge_section = PromptSection(
            "Second parent solution (Approach B) — MERGE OBJECTIVE",
            "This run merges two independently evolved optimizer branches. Approach A is "
            "the incumbent lineage shown above as parent. Below is the best program of the "
            f"other branch ({cfg.merge_from}). Your goal: COMBINE the two — take the "
            "strongest mechanisms of each (seed/basin choice, feasibility-margin "
            "management, step/momentum logic, budget allocation) and integrate them into "
            "one coherent better optimizer. Do not concatenate blindly; when mechanisms "
            "conflict, keep the one with better evidence and say so in your reasoning.\n\n"
            "```python\n" + merge_src[:15000] + "\n```")
    getattr(ax["feedback"], "bind", lambda *a: None)(run_dir, ledger)
    getattr(ax["knowledge"], "bind", lambda *a: None)(
        task, llm, ax["roles"], cfg.temperatures, ledger)
    ledger.append({"type": "run_start", "config": cfg.to_dict(), "run_dir": str(run_dir)})

    pool = Pool()
    seed_src, seed_meta = task.seed_code(), {"gen": 0, "seed": True}
    if cfg.resume_from:
        seed_src = resume_code(cfg.resume_from, cfg.task)
        seed_meta["resume_from"] = cfg.resume_from
    seed_cand = Candidate(code=seed_src, id="c0000", meta=seed_meta)
    _evaluate_into(task, seed_cand)
    ax["gate"].accept(seed_cand, pool)  # holdout gate scores val here; a broken seed is a task bug
    _probe_extra_splits(task, seed_cand)
    ax["search"].insert(pool, seed_cand)
    ledger.append({"type": "candidate", "gen": 0, "accepted": True, **seed_cand.to_dict()})

    last, stop_reason, failures = seed_cand, "generations_exhausted", 0
    best_comb, stall_n, n_cands = _combined(seed_cand.scores), 0, 0
    refine_window: list[Candidate] = []

    def admit(cand) -> bool:
        """Evaluate + gate + ledger one non-writer candidate (analyst injection,
        refiner v2), with the same stall accounting as the writer path."""
        nonlocal last, n_cands, best_comb, stall_n
        _evaluate_into(task, cand)
        cand.cost = {"usd": round(guard.usd, 4), "calls": guard.calls}
        acc = getattr(ax["search"], "adopt", lambda c: False)(cand) \
            or ax["gate"].accept(cand, pool)
        if acc:
            _probe_extra_splits(task, cand)
            ax["search"].insert(pool, cand)
        ledger.append({"type": "candidate", "accepted": acc,
                       "elapsed": round(guard.elapsed(), 1), **cand.to_dict()})
        last = cand
        n_cands += 1
        comb = _combined(cand.scores)
        if comb > best_comb:
            best_comb, stall_n = comb, 0
        else:
            stall_n += 1
        getattr(ax["knowledge"], "observe", lambda *a: None)(pool, cand, acc)
        return acc

    def consume_inject(gen: int) -> None:
        """Analyst-written candidate modules (<run_dir>/INJECT/*.py): evaluated
        and gated like writer candidates, then moved to INJECT/used/."""
        inj = run_dir / "INJECT"
        if not inj.is_dir():
            return
        for k, f in enumerate(sorted(inj.glob("*.py"))[:4]):
            src = f.read_text()
            (inj / "used").mkdir(exist_ok=True)
            f.rename(inj / "used" / f.name)
            head = {m.group(1).lower(): m.group(2).strip() for m in re.finditer(
                r"^#\s*(Idea|Prediction):\s*(.+)$", src[:2000], re.M | re.I)}
            admit(Candidate(
                code=src, id=f"c{gen:04d}i{k}",
                parent_id=pool.best("train").id if pool.all else None,
                meta={"gen": gen, "injected": f.name,
                      "idea": head.get("idea"), "prediction": head.get("prediction")}))

    for gen in range(1, cfg.generations + 1):
        if (run_dir / "STOP").exists():
            stop_reason = "user_stop"
            break
        try:
            consume_inject(gen)
        except Exception as e:  # a broken injected file must not kill the run
            ledger.append({"type": "gen_error", "gen": gen,
                           "error": f"inject: {str(e)[:280]}"})
        if cfg.stall_stop and stall_n >= cfg.stall_stop:
            stop_reason = f"stall: {stall_n} candidates without a new best"
            break
        try:
            guard.check()
            parents = ax["search"].select_parents(pool, rng)
            hint = getattr(ax["feedback"], "long_term", "") or str(last.meta.get("error", ""))
            sections = (
                ax["knowledge"].prompt_sections(task, hint)
                + ax["feedback"].build_context(pool, parents, last)
                + getattr(ax["search"], "prompt_sections", lambda: [])()
            )
            if merge_section is not None:
                sections.append(merge_section)
            hint_file = run_dir / "HINT"  # live user steering: edit/delete anytime
            if hint_file.exists() and hint_file.read_text().strip():
                sections.append(PromptSection(
                    "Directive from the user", hint_file.read_text().strip()[:2000]))
            prompt = "\n\n".join(
                [f"# Task\n{task.description}"]
                + [s.render() for s in sections]
                + ["Write an improved version of the function. Keep the exact signature. "
                   "Start with one line `Idea: <one-sentence summary of your change>`, "
                   "then one line `Prediction: <expected effect and mechanism>`, "
                   "then brief reasoning, then exactly one fenced code block."]
            )
            tools, handlers = [], {}
            for axobj in (ax["knowledge"], ax["feedback"]):
                for t in getattr(axobj, "tools", lambda: [])():
                    tools.append(t)
                    handlers[t["function"]["name"]] = axobj.call_tool
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}]
            for attempt in range(COMPILE_REPAIRS + 1):
                text = llm.chat(
                    ax["roles"].writer(), messages,
                    temperature=cfg.temperatures["writer"], role="writer",
                    tools=tools or None,
                    tool_handler=(lambda n, a: handlers[n](n, a)
                                  if n in handlers else f"unknown tool {n}") if tools else None,
                )
                code = parse_code(text)
                cand = Candidate(code=code or "", parent_id=parents[0].id,
                                 id=f"c{gen:04d}" + (f"r{attempt}" if attempt else ""),
                                 meta={"gen": gen, "idea": parse_idea(text),
                                       "prediction": parse_prediction(text),
                                       "reasoning": parse_reasoning(text)})
                if attempt:
                    cand.meta["repair"] = attempt
                if code is None:
                    cand.scores["train"] = float("-inf")
                    cand.meta["error"] = "parse_error"
                elif not cand.meta.get("idea") and attempt == 0:
                    # contract violation (§6.4): one repair round, and don't
                    # spend an expensive eval on an unlabeled candidate
                    cand.scores["train"] = float("-inf")
                    cand.meta["error"] = "missing_idea"
                else:
                    _evaluate_into(task, cand)
                err = str(cand.meta.get("error", ""))
                if attempt == COMPILE_REPAIRS or not (
                        _compile_error(err) or err in ("parse_error", "missing_idea")):
                    break
                cand.cost = {"usd": round(guard.usd, 4), "calls": guard.calls}
                ledger.append({"type": "candidate", "accepted": False,
                               "elapsed": round(guard.elapsed(), 1), **cand.to_dict()})
                guard.check()
                if _compile_error(err):
                    followup = (
                        "Your code failed to compile:\n" + err[:1500] +
                        "\nFix the compile error and output the complete corrected "
                        "code in one fenced code block.")
                elif err == "missing_idea":
                    followup = (
                        "Your response was missing the required `Idea:` line, so it "
                        "was not evaluated. Resend your complete answer now: one "
                        "line `Idea: <one-sentence summary>`, one line "
                        "`Prediction: <expected effect and mechanism>`, then "
                        "exactly one fenced code block with the full code.")
                else:
                    followup = (
                        "Your response contained no fenced code block, so it could "
                        "not be evaluated. Output the complete improved function "
                        "now — `Idea:` line, `Prediction:` line, then exactly one "
                        "fenced code block.")
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": followup}]
            failures = 0
        except BudgetExceeded as e:
            stop_reason = f"budget: {e}"
            break
        except Exception as e:  # persistent LLM failure — skip the generation, give up after 3
            failures += 1
            ledger.append({"type": "gen_error", "gen": gen, "error": str(e)[:300]})
            if failures >= 3:
                stop_reason = f"llm_error: {e}"
                break
            continue
        cand.cost = {"usd": round(guard.usd, 4), "calls": guard.calls}
        cand.meta["novelty"] = round(
            1 - difflib.SequenceMatcher(None, parents[0].code, cand.code).ratio(), 3)
        accepted = getattr(ax["search"], "adopt", lambda c: False)(cand) \
            or ax["gate"].accept(cand, pool)
        if accepted:
            _probe_extra_splits(task, cand)
            ax["search"].insert(pool, cand)
        ledger.append({
            "type": "candidate", "accepted": accepted,
            "elapsed": round(guard.elapsed(), 1), **cand.to_dict(),
        })
        last = cand
        n_cands += 1
        comb = _combined(cand.scores)
        if comb > best_comb:
            best_comb, stall_n = comb, 0
        else:
            stall_n += 1
        getattr(ax["knowledge"], "observe", lambda *a: None)(pool, cand, accepted)
        if refiner is not None and cand.code:
            # windowed v2 pass: collect writer candidates; every refiner_every of
            # them one Opus session digests the whole window (+ wiki search) and
            # synthesizes ONE refinement, evaluated + gated like any candidate.
            # note() digests each windowed candidate now, since a v2 displacing
            # it as `last` would otherwise skip its feedback digest.
            getattr(ax["feedback"], "note", lambda *a: None)(cand, pool)
            refine_window.append(cand)
            if len(refine_window) >= cfg.refiner_every:
                window, refine_window = refine_window, []
                v2 = refiner.refine(task, pool, window, gen)
                if v2 is not None:
                    v2.meta["novelty"] = round(1 - difflib.SequenceMatcher(
                        None, pool.best("train").code, v2.code).ratio(), 3)
                    admit(v2)
        if analyst is not None:
            analyst.maybe(pool, n_cands, getattr(ax["feedback"], "recent", []))
        if cfg.stall_stop and stall_n >= cfg.stall_stop:
            stop_reason = f"stall: {stall_n} candidates without a new best"
            break
        if gen % PUBLIC_EVERY == 0:
            _report_public(task, ledger, pool.best("train"), gen, guard)

    best = max(pool.all, key=lambda c: _combined(c.scores))
    evals, priv = _private_median(task, best.code)
    best.scores["private"] = priv.score
    (run_dir / "best.py").write_text(best.code)
    summary = {
        "type": "run_end", "stop_reason": stop_reason, "best_id": best.id,
        "train": best.score("train"), "val": best.scores.get("val"), "private": priv.score,
        "private_metrics": priv.metrics,
        "generalization_gap": round(priv.score - best.score("train"), 6),
        "private_all": [round(r.score, 4) for r in evals],
        "usd": round(guard.usd, 4), "calls": guard.calls,
        "seconds": round(guard.elapsed(), 1),
        "render": task.render(best.code, priv),
    }
    # second Pareto champion: best on the long-budget probe, if it's a different program
    with30 = [c for c in pool.all if "val30" in c.scores]
    best30 = max(with30, key=lambda c: c.scores["val30"]) if with30 else None
    if best30 is not None and best30.id != best.id:
        evals30, p30 = _private_median(task, best30.code)
        (run_dir / "best30.py").write_text(best30.code)
        summary.update(best30_id=best30.id, best30_val30=round(best30.scores["val30"], 4),
                       best30_private=p30.score,
                       best30_private_all=[round(r.score, 4) for r in evals30])
    ledger.append(summary)
    return summary
