"""The evolution loop: generate → evaluate → gate → insert → repeat.

Composes the 5 axis objects from the config and never special-cases a switch.
A run is a pure function of (config, task, seed) modulo LLM nondeterminism —
which is why every event lands in the ledger.
"""
from __future__ import annotations

import difflib
import importlib
import random
import time
from pathlib import Path

from axes.feedback import Reflections, ScoreOnly
from axes.gate import HoldoutGate, PublicOnly
from axes.knowledge import Off, WikiFS
from axes.memory import MemoryWiki
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
    "knowledge": {"off": Off, "wiki_fs": WikiFS},
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
    Raises ValueError so the API can reject bad requests before a run starts."""
    d = _ROOT / "runs" / Path(run_id).name
    if not (d / "ledger.jsonl").exists():
        raise ValueError(f"resume_from: no ledger for run {run_id!r}")
    best, score = "", float("-inf")
    for ev in Ledger(d).read():
        if ev["type"] == "run_start" and ev["config"].get("task") != task_name:
            raise ValueError(f"resume_from: {run_id!r} is task "
                             f"{ev['config'].get('task')!r}, not {task_name!r}")
        if ev["type"] == "candidate" and ev.get("accepted") and ev.get("code") \
                and _combined(ev.get("scores", {})) >= score:
            best, score = ev["code"], _combined(ev["scores"])
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


def run(cfg: Config, run_dir: str | Path | None = None, llm_factory=LLM) -> dict:
    task = load_task(cfg.task)
    run_dir = Path(run_dir or _ROOT / "runs" / f"{cfg.task}-{cfg.seed}-{int(time.time())}")
    ledger = Ledger(run_dir)
    guard = BudgetGuard(**cfg.budget)
    llm = llm_factory(ledger, guard)
    rng = random.Random(cfg.seed)
    ax = assemble(cfg, task, llm)
    getattr(ax["feedback"], "bind", lambda *a: None)(run_dir, ledger)
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
    for gen in range(1, cfg.generations + 1):
        if (run_dir / "STOP").exists():
            stop_reason = "user_stop"
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
                else:
                    _evaluate_into(task, cand)
                err = str(cand.meta.get("error", ""))
                if attempt == COMPILE_REPAIRS or not (_compile_error(err) or err == "parse_error"):
                    break
                cand.cost = {"usd": round(guard.usd, 4), "calls": guard.calls}
                ledger.append({"type": "candidate", "accepted": False,
                               "elapsed": round(guard.elapsed(), 1), **cand.to_dict()})
                guard.check()
                followup = (
                    "Your code failed to compile:\n" + err[:1500] +
                    "\nFix the compile error and output the complete corrected "
                    "code in one fenced code block."
                ) if _compile_error(err) else (
                    "Your response contained no fenced code block, so it could not "
                    "be evaluated. Output the complete improved function now — "
                    "`Idea:` line, `Prediction:` line, then exactly one fenced "
                    "code block.")
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
        if gen % PUBLIC_EVERY == 0:
            _report_public(task, ledger, pool.best("train"), gen, guard)

    best = max(pool.all, key=lambda c: _combined(c.scores))
    # median of 3: single anytime private evals spread ~0.1-0.18 gap-pct pts, and a
    # transient sandbox failure must not report -inf for a working candidate
    evals = [task.evaluate(best.code, "private") for _ in range(3)]
    ok = [r for r in evals if r.score > float("-inf")] or evals
    priv = sorted(ok, key=lambda r: r.score)[len(ok) // 2]
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
        evals30 = [task.evaluate(best30.code, "private") for _ in range(3)]
        ok30 = [r for r in evals30 if r.score > float("-inf")] or evals30
        p30 = sorted(ok30, key=lambda r: r.score)[len(ok30) // 2]
        (run_dir / "best30.py").write_text(best30.code)
        summary.update(best30_id=best30.id, best30_val30=round(best30.scores["val30"], 4),
                       best30_private=p30.score,
                       best30_private_all=[round(r.score, 4) for r in evals30])
    ledger.append(summary)
    return summary
