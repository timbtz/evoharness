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

SYSTEM = (
    "You are an expert algorithm designer evolving a Python function. "
    "Respond with exactly one ```python code block containing the complete improved function "
    "(imports included). Brief reasoning before the block is fine."
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
    seed_cand = Candidate(code=task.seed_code(), id="c0000", meta={"gen": 0, "seed": True})
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
                   "Output exactly one ```python code block."]
            )
            tools = ax["knowledge"].tools()
            text = llm.chat(
                ax["roles"].writer(),
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=cfg.temperatures["writer"], role="writer",
                tools=tools or None,
                tool_handler=ax["knowledge"].call_tool if tools else None,
            )
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

        code = parse_code(text)
        cand = Candidate(code=code or "", parent_id=parents[0].id, id=f"c{gen:04d}",
                         meta={"gen": gen})
        if code is None:
            cand.scores["train"] = float("-inf")
            cand.meta["error"] = "parse_error"
        else:
            _evaluate_into(task, cand)
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
