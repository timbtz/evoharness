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
from axes.roles import SingleStrong, SplitRoles
from axes.search import Greedy11, Islands
from core.candidate import Candidate, Pool
from core.config import Config
from core.ledger import BudgetExceeded, BudgetGuard, Ledger
from core.llm import LLM, parse_code

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
    "Respond with exactly one fenced code block containing the complete improved function "
    "and anything it needs (imports/includes/helpers). Brief reasoning before the block is fine."
)

_AXES = {
    "feedback": {"score_only": ScoreOnly, "reflections": Reflections},
    "gate": {"public_only": PublicOnly, "holdout": HoldoutGate},
    "search": {"greedy": Greedy11, "islands": Islands},
    "knowledge": {"off": Off, "wiki_fs": WikiFS},
    "roles": {"single_strong": SingleStrong, "split_roles": SplitRoles},
}


def load_task(name: str):
    return importlib.import_module(f"tasks.{name}.task").TASK


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
                and ev.get("scores", {}).get("train", float("-inf")) >= score:
            best, score = ev["code"], ev["scores"]["train"]
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
    ledger.append({"type": "run_start", "config": cfg.to_dict(), "run_dir": str(run_dir)})

    pool = Pool()
    seed_src, seed_meta = task.seed_code(), {"gen": 0, "seed": True}
    if cfg.resume_from:
        seed_src = resume_code(cfg.resume_from, cfg.task)
        seed_meta["resume_from"] = cfg.resume_from
    seed_cand = Candidate(code=seed_src, id="c0000", meta=seed_meta)
    _evaluate_into(task, seed_cand)
    ax["gate"].accept(seed_cand, pool)  # holdout gate scores val here; a broken seed is a task bug
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
            )
            prompt = "\n\n".join(
                [f"# Task\n{task.description}"]
                + [s.render() for s in sections]
                + ["Write an improved version of the function. Keep the exact signature. "
                   "Output exactly one fenced code block."]
            )
            tools = ax["knowledge"].tools()
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt}]
            for attempt in range(COMPILE_REPAIRS + 1):
                text = llm.chat(
                    ax["roles"].writer(), messages,
                    temperature=cfg.temperatures["writer"], role="writer",
                    tools=tools or None,
                    tool_handler=ax["knowledge"].call_tool if tools else None,
                )
                code = parse_code(text)
                cand = Candidate(code=code or "", parent_id=parents[0].id,
                                 id=f"c{gen:04d}" + (f"r{attempt}" if attempt else ""),
                                 meta={"gen": gen})
                if attempt:
                    cand.meta["repair"] = attempt
                if code is None:
                    cand.scores["train"] = float("-inf")
                    cand.meta["error"] = "parse_error"
                else:
                    _evaluate_into(task, cand)
                err = str(cand.meta.get("error", ""))
                if attempt == COMPILE_REPAIRS or not _compile_error(err):
                    break
                cand.cost = {"usd": round(guard.usd, 4), "calls": guard.calls}
                ledger.append({"type": "candidate", "accepted": False,
                               "elapsed": round(guard.elapsed(), 1), **cand.to_dict()})
                guard.check()
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content":
                        "Your code failed to compile:\n" + err[:1500] +
                        "\nFix the compile error and output the complete corrected "
                        "code in one fenced code block."}]
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
        accepted = ax["gate"].accept(cand, pool)
        if accepted:
            ax["search"].insert(pool, cand)
        ledger.append({
            "type": "candidate", "accepted": accepted,
            "elapsed": round(guard.elapsed(), 1), **cand.to_dict(),
        })
        last = cand
        if gen % PUBLIC_EVERY == 0:
            _report_public(task, ledger, pool.best("train"), gen, guard)

    best = pool.best("train")
    priv = task.evaluate(best.code, "private")
    best.scores["private"] = priv.score
    (run_dir / "best.py").write_text(best.code)
    summary = {
        "type": "run_end", "stop_reason": stop_reason, "best_id": best.id,
        "train": best.score("train"), "private": priv.score,
        "private_metrics": priv.metrics,
        "generalization_gap": round(priv.score - best.score("train"), 6),
        "usd": round(guard.usd, 4), "calls": guard.calls,
        "seconds": round(guard.elapsed(), 1),
        "render": task.render(best.code, priv),
    }
    ledger.append(summary)
    return summary
