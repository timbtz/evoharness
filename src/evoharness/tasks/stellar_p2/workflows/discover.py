"""One-model, persistent, web-enabled Stellar P2 discovery loop.

The campaign deliberately starts without an inherited candidate.  GLM-5.2
bootstraps the first independent optimizer, receives the trusted physics result,
and then improves or replaces candidates from its own lineage.  All model text,
candidate programs, web-source URLs, measurements, and the incumbent survive a
process restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from evoharness.paths import RUNS_DIR

from evoharness.strategies.research import fetch_url, web_search
from evoharness.engine.campaign_lock import CampaignLock
from evoharness.engine.ledger import BudgetExceeded, BudgetGuard, Ledger
from evoharness.engine.llm import LLM, parse_code, parse_idea, parse_prediction, parse_reasoning
from evoharness.tasks.stellar_p2.discovery.boundaries import enrich_metrics, stable_hash
from evoharness.tasks.stellar_p2.workflows.evaluate import (_evaluator_state, _novelty_record,
                                                _signed, audit_candidate)
from evoharness.tasks.stellar_p2.discovery.census import atomic_json


MODEL = "glm-5.2"
SCHEMA_VERSION = 1
WEB_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the open web for research, algorithms, and cross-domain analogies.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch one public HTTP(S) page as plain text after finding it in search.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}}, "required": ["url"]}}},
]

CONTINUE_PROMPT = """Do not stop. Believe that another materially different candidate is
possible and continue the search. If the direct approach is exhausted, combine mechanisms from
other research areas or engineering domains. Use web_search and fetch_url to find a fresh analogy,
then turn it into a concrete, testable optimizer. Return a new complete candidate under the exact
Idea/Prediction/Memory/code contract; do not return a plan, refusal, or stopping recommendation."""

SYSTEM = """You are the sole model in a persistent scientific code-improvement loop. You design
independent Stellar P2 optimizer programs and learn from their measured physics results. You never
decide that the campaign is finished: only the host's hard time, call, and dollar limits may stop
it. When stuck, remain confident, search for new mechanisms, cross-pollinate ideas from other
fields, and try a materially different candidate.

Public seed-bank access, inherited boundaries, embedded boundary coefficient dumps, filesystem
access, and network access from candidate code are forbidden. Web tools are available to YOU for
research; candidate programs themselves remain sandboxed and offline.

Every answer must contain, in this order:
Idea: <one sentence describing the genuinely new mechanism>
Prediction: <falsifiable expected metric changes>
Memory: <one concise lesson that should persist if this attempt fails or succeeds>
<brief reasoning and source URLs used>
```python
<one complete module defining solve(fm, rng)>
```
Never omit the candidate code and never repeat an identical program."""

STOP_PATTERNS = re.compile(
    r"\b(?:i (?:will|should|must|need to) stop|we should stop|the campaign should stop|"
    r"give up|cannot (?:continue|improve)|no (?:viable|further) "
    r"(?:candidate|improvement)|exhausted (?:the )?(?:search|ideas))\b", re.I)


def parse_memory(text: str) -> str | None:
    match = re.search(r"^\s*\**Memory\**\s*[:*]\s*(.+)$", text, re.M)
    return match.group(1).strip().strip("*") if match else None


def response_requests_stop(text: str) -> bool:
    """Detect a model trying to terminate; missing code is also a stop-equivalent."""
    return parse_code(text) is None or bool(STOP_PATTERNS.search(text))


def _metric_summary(attempt: dict[str, Any]) -> dict[str, Any]:
    metrics = attempt.get("metrics") or {}
    return {
        "round": attempt.get("round"),
        "status": attempt.get("status"),
        "accepted": attempt.get("accepted", False),
        "idea": attempt.get("idea"),
        "prediction": attempt.get("prediction"),
        "memory": attempt.get("memory"),
        "selection_score": attempt.get("selection_score"),
        "objective_L": metrics.get("objective_L"),
        "feasibility": metrics.get("feasibility"),
        "active_violation": metrics.get("active_violation"),
        "honest_score": metrics.get("honest_score"),
        "error": attempt.get("error"),
        "web_sources": attempt.get("web_sources", [])[:5],
    }


def memory_markdown(state: dict[str, Any]) -> str:
    """Human-readable durable memory; state.json remains the canonical record."""
    attempts = state.get("attempts", [])
    best = next((row for row in attempts
                 if row.get("candidate_hash") == state.get("best_candidate_hash")), None)
    lines = ["# Persistent GLM-5.2 research memory", "",
             "This campaign starts without an inherited candidate. Its incumbent is always from",
             "its own bank-disabled lineage; every measurement below came from the host evaluator.", ""]
    if best:
        lines.extend(["## Current incumbent", "", json.dumps(
            _metric_summary(best), indent=2, sort_keys=True), ""])
    lines.extend(["## Attempts", ""])
    for row in attempts:
        lines.extend([f"### Round {row.get('round')}", "",
                      json.dumps(_metric_summary(row), indent=2, sort_keys=True), ""])
    return "\n".join(lines).rstrip() + "\n"


def _memory_digest(state: dict[str, Any], maximum_chars: int = 18_000) -> str:
    attempts = state.get("attempts", [])
    failures: dict[str, int] = {}
    for row in attempts:
        key = str((row.get("metrics") or {}).get("active_violation") or
                  row.get("status") or "unknown")
        failures[key] = failures.get(key, 0) + 1
    payload = {
        "best_candidate_hash": state.get("best_candidate_hash"),
        "best_selection_score": state.get("best_selection_score"),
        "outcome_counts": failures,
        "recent_attempts": [_metric_summary(row) for row in attempts[-16:]],
    }
    return json.dumps(payload, indent=2, sort_keys=True)[-maximum_chars:]


STEERING_PATH: Path | None = None  # set by main(); <run_dir>/STEERING.md


def _steering_section() -> str:
    """Optional host steering: if <run_dir>/STEERING.md exists, its text is
    injected verbatim into every round's prompt (edit/delete anytime; read per
    round, no restart needed once the process runs a build with this hook)."""
    if STEERING_PATH is None or not STEERING_PATH.exists():
        return ""
    text = STEERING_PATH.read_text().strip()
    if not text:
        return ""
    return f"\n## Directive from the host operator\n{text[:6000]}\n"


def build_prompt(task_description: str, state: dict[str, Any], incumbent_code: str | None,
                 round_number: int, runtime_contract: str = "") -> str:
    lineage = ("There is no candidate yet. Start from a null point using only the documented "
               "independent constructors and formulate the first complete optimizer."
               if incumbent_code is None else
               "The incumbent below was created by this same campaign, not inherited. Improve it "
               "or replace it with a genuinely different construction; optimize measured score, "
               "not rhetorical novelty.")
    incumbent = incumbent_code or "(none — null bootstrap round)"
    return f"""# Round {round_number}

{lineage}

Before proposing code, use web_search at least once. Search outside the immediate stellarator
literature as well as within it: useful analogies may come from continuation methods, constraint
handling, control, experimental design, numerical optimization, geometry, evolutionary search, or
another domain. Fetch promising primary or technical sources. Cite the URLs in your reasoning and
translate the mechanism rather than copying source code.

{_steering_section()}
Treat the persistent measurements as ground truth. Do not repeat a failed mechanism with cosmetic
parameter changes. A new candidate should either improve the incumbent's selection score or teach
a sharply falsifiable lesson that opens a different route. You are not allowed to answer that the
search should stop; if you feel stuck, follow the confidence-and-cross-domain instruction in the
system message and keep trying.

## Persistent measured memory
{_memory_digest(state)}

## Current self-created incumbent code
```python
{incumbent}
```

## Executable task contract
IMPORTANT OVERRIDE: the public seed bank is technically disabled and calling seed_bank or
seed_bank_info is rejected before evaluation. Do not rely on any inherited boundary.
{runtime_contract}

{task_description}
"""


def continuation_prompt(problem: str) -> str:
    return f"""{CONTINUE_PROMPT}

The previous response could not enter the evaluator for this reason:
{problem[:1600]}
"""


def _initial_state(settings: dict[str, Any], evaluator_fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "arm": "independent",
        "bank_disabled": True,
        "created_at": time.time(),
        "settings": settings,
        "evaluator_fingerprint": evaluator_fingerprint,
        "round": 0,
        "attempts": [],
        "best_candidate_hash": None,
        "best_candidate_path": None,
        "best_selection_score": None,
    }


def _load_state(run_dir: Path, settings: dict[str, Any],
                evaluator_fingerprint: str) -> dict[str, Any]:
    path = run_dir / "state.json"
    if not path.exists():
        return _initial_state(settings, evaluator_fingerprint)
    state = json.loads(path.read_text())
    if (state.get("schema_version") != SCHEMA_VERSION or state.get("model") != MODEL or
            state.get("arm") != "independent" or not state.get("bank_disabled")):
        raise RuntimeError("incompatible campaign state; use a new run directory")
    if state.get("settings") != settings:
        raise RuntimeError("evaluator settings changed; use a new run directory")
    if state.get("evaluator_fingerprint") != evaluator_fingerprint:
        raise RuntimeError("evaluator fingerprint changed; use a new run directory")
    return state


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    atomic_json(run_dir / "state.json", state)
    (run_dir / "memory.md").write_text(memory_markdown(state))


def _tool_handler(sources: list[str], queries: list[str]):
    def handle(name: str, args: dict) -> str:
        try:
            if name == "web_search":
                query = str(args.get("query", "")).strip()
                queries.append(query)
                rows = web_search(query, count=8)
                sources.extend(str(row.get("url", "")) for row in rows if row.get("url"))
                return "\n\n".join(
                    f"{row.get('title', '')}\n{row.get('url', '')}\n{row.get('snippet', '')}"
                    for row in rows) or "no results"
            if name == "fetch_url":
                url = str(args.get("url", "")).strip()
                sources.append(url)
                return fetch_url(url)
            return f"unknown tool: {name}"
        except Exception as exc:
            return f"web tool failed: {type(exc).__name__}: {str(exc)[:300]}"
    return handle


def _generate(llm: LLM, task_description: str, state: dict[str, Any],
              incumbent_code: str | None, round_number: int,
              known_hashes: set[str]) -> tuple[str | None, dict[str, Any]]:
    sources: list[str] = []
    queries: list[str] = []
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_prompt(
            task_description, state, incumbent_code, round_number)},
    ]
    last_text = ""
    problem = ""
    for attempt in range(3):
        if attempt:
            messages.extend([
                {"role": "assistant", "content": last_text[-12_000:]},
                {"role": "user", "content": continuation_prompt(problem)},
            ])
        last_text = llm.chat(
            MODEL, messages, temperature=.75 if attempt == 0 else .45,
            role="persistent_candidate" if attempt == 0 else "persistent_continue",
            max_tokens=32_768, tools=WEB_TOOLS,
            tool_handler=_tool_handler(sources, queries), rounds=8)
        code = parse_code(last_text)
        if code is None:
            problem = ("You tried to stop or returned no fenced candidate. "
                       "The host requires a complete new solve(fm, rng) module.")
            continue
        if response_requests_stop(last_text):
            problem = ("You supplied code but also tried to terminate the search. "
                       "Continue confidently with a materially different candidate.")
            continue
        idea, prediction, lesson = (parse_idea(last_text), parse_prediction(last_text),
                                    parse_memory(last_text))
        if not all((idea, prediction, lesson)):
            problem = "The response omitted one or more required Idea, Prediction, or Memory lines."
            continue
        errors = audit_candidate(code)
        digest = hashlib.sha256(code.encode()).hexdigest()
        if digest in known_hashes:
            problem = "The program is byte-for-byte identical to an earlier candidate."
            continue
        if errors:
            problem = "Candidate audit failed: " + "; ".join(errors)
            continue
        return code, {
            "idea": idea,
            "prediction": prediction,
            "memory": lesson,
            "reasoning": parse_reasoning(last_text)[:12_000],
            "tried_to_stop": response_requests_stop(last_text),
            "web_queries": list(dict.fromkeys(q for q in queries if q))[:20],
            "web_sources": list(dict.fromkeys(s for s in sources if s))[:40],
            "generation_attempts": attempt + 1,
        }
    return None, {
        "idea": parse_idea(last_text), "prediction": parse_prediction(last_text),
        "memory": parse_memory(last_text), "reasoning": parse_reasoning(last_text)[:12_000],
        "tried_to_stop": True, "error": problem,
        "web_queries": list(dict.fromkeys(q for q in queries if q))[:20],
        "web_sources": list(dict.fromkeys(s for s in sources if s))[:40],
        "generation_attempts": 3,
    }


def _evaluate(task, code: str, run_dir: Path, round_number: int,
              evaluator_fingerprint: str, execution_fingerprint: str,
              evaluator_spec: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(code.encode()).hexdigest()
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"c{round_number:04d}-{digest[:12]}.py"
    candidate_path.write_text(code.rstrip() + "\n")
    started = time.time()
    with CampaignLock(RUNS_DIR / ".stellar-physics.lock"):
        result = task.evaluate(code, "train")
    metrics = enrich_metrics(result.metrics)
    record: dict[str, Any] = {
        "schema_version": 2,
        "arm": "independent",
        "bank_disabled": True,
        "round": round_number,
        "candidate_hash": digest,
        "candidate_path": str(candidate_path.relative_to(run_dir)),
        "candidate_source": code,
        "evaluator_fingerprint": evaluator_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "evaluator_spec": evaluator_spec,
        "fidelity": "very_low_fidelity",
        "started_at": started,
        "finished_at": time.time(),
        "seconds": result.seconds,
        "error": result.error,
        "selection_score": result.score,
        "metrics": metrics,
    }
    if not result.error:
        cache_key = hashlib.sha1(code.encode()).hexdigest()[:12]
        boundary = task._bcache[cache_key]
        record.update(boundary=boundary, boundary_hash=stable_hash(boundary),
                      novelty_audit=_novelty_record(boundary))
    record = _signed(record, run_dir)
    artifact = run_dir / "evaluations" / digest / "very_low_fidelity.json"
    atomic_json(artifact, record)
    record["evaluation_path"] = str(artifact.relative_to(run_dir))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=RUNS_DIR / "stellar-zai-persistent-1")
    parser.add_argument("--rounds", type=int, default=1000,
                        help="maximum new proposal/evaluation rounds in this invocation")
    parser.add_argument("--max-usd", type=float, default=50.0,
                        help="cumulative model budget for this persistent campaign")
    parser.add_argument("--max-calls", type=int, default=3000,
                        help="cumulative model-call limit including tool rounds and retries")
    parser.add_argument("--walltime-hours", type=float, default=24.0,
                        help="walltime for this invocation; rerun to resume persistent memory")
    parser.add_argument("--max-evals", type=int, default=24,
                        help="metered physics evaluations available to each candidate")
    parser.add_argument("--candidate-cpu-seconds", type=float, default=600.0)
    parser.add_argument("--eval-timeout", type=float, default=150.0)
    parser.add_argument("--print-prompt", action="store_true",
                        help="initialize state and print the next prompt without calling the model")
    args = parser.parse_args()
    if (args.rounds < 1 or args.max_usd <= 0 or args.max_calls < 1 or
            args.walltime_hours <= 0 or not 1 <= args.max_evals <= 400 or
            not 60 <= args.candidate_cpu_seconds <= 7200 or
            not 30 <= args.eval_timeout <= 1800):
        parser.error("invalid campaign or evaluator budget")

    # This must precede the task import: it is a technical capability boundary,
    # not merely a model instruction.
    os.environ["STELLAR_NO_BANK"] = "1"
    evaluator_settings = {
        "max_evals": args.max_evals,
        "cpu_budget": args.candidate_cpu_seconds,
        "collect_top": 3,
        "workers": 1,
        "eval_timeout": args.eval_timeout,
    }
    os.environ["STELLAR_TRAIN_OVERRIDES"] = json.dumps(evaluator_settings)
    from evoharness.tasks.stellar_p2.task import TASK

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    global STEERING_PATH
    STEERING_PATH = run_dir / "STEERING.md"
    evaluator_fingerprint, execution_fingerprint, evaluator_spec = _evaluator_state(TASK)
    state = _load_state(run_dir, evaluator_settings, evaluator_fingerprint)
    _save_state(run_dir, state)
    runtime_contract = ("\nHOST RUNTIME BUDGET (overrides any default numbers below): each "
                        f"candidate has {args.max_evals} metered evaluations, "
                        f"{args.candidate_cpu_seconds:g} CPU-seconds, one worker, and a "
                        f"{args.eval_timeout:g}-second per-evaluation timeout.\n")
    task_description = runtime_contract + "\n" + TASK.description
    incumbent_path = state.get("best_candidate_path")
    incumbent_code = ((run_dir / incumbent_path).read_text()
                      if incumbent_path and (run_dir / incumbent_path).is_file() else None)
    if args.print_prompt:
        print(build_prompt(task_description, state, incumbent_code,
                           int(state.get("round", 0)) + 1))
        return 0

    ledger = Ledger(run_dir / "zai")
    guard = BudgetGuard(args.max_usd, args.max_calls, args.walltime_hours * 3600)
    previous = ledger.read()
    guard.calls = sum(row.get("type") in {"llm_call", "llm_error"} for row in previous)
    guard.usd = sum(float(row.get("usd", 0.0) or 0.0) for row in previous
                    if row.get("type") == "llm_call")
    llm = LLM(ledger, guard)
    known_hashes = {str(row.get("candidate_hash")) for row in state["attempts"]
                    if row.get("candidate_hash")}
    invocation_rounds = 0
    halt_reason = None

    while invocation_rounds < args.rounds:
        round_number = int(state.get("round", 0)) + 1
        try:
            code, proposal = _generate(llm, task_description, state, incumbent_code,
                                       round_number, known_hashes)
        except BudgetExceeded as exc:
            halt_reason = f"model budget: {exc}"
            break
        except KeyboardInterrupt:
            halt_reason = "operator interrupt"
            break
        except Exception as exc:
            proposal = {"status": "model_failed", "error": str(exc)[:1000]}
            code = None

        state["round"] = round_number
        invocation_rounds += 1
        if code is None:
            attempt = {"round": round_number, "status": "proposal_failed",
                       "accepted": False, **proposal}
        else:
            try:
                evaluation = _evaluate(
                    TASK, code, run_dir, round_number, evaluator_fingerprint,
                    execution_fingerprint, evaluator_spec)
                score = float(evaluation["selection_score"])
                old_score = state.get("best_selection_score")
                accepted = math.isfinite(score) and (
                    old_score is None or score > float(old_score))
                attempt = {
                    "round": round_number,
                    "status": "evaluated" if not evaluation.get("error") else "evaluation_failed",
                    "accepted": accepted,
                    "candidate_hash": evaluation["candidate_hash"],
                    "candidate_path": evaluation["candidate_path"],
                    "evaluation_path": evaluation["evaluation_path"],
                    "selection_score": score,
                    "metrics": evaluation["metrics"],
                    "error": evaluation.get("error"),
                    **proposal,
                }
                known_hashes.add(evaluation["candidate_hash"])
                if accepted:
                    state["best_candidate_hash"] = evaluation["candidate_hash"]
                    state["best_candidate_path"] = evaluation["candidate_path"]
                    state["best_selection_score"] = score
                    incumbent_code = code
                    (run_dir / "best.py").write_text(code.rstrip() + "\n")
            except KeyboardInterrupt:
                halt_reason = "operator interrupt during evaluation"
                break
            except Exception as exc:
                attempt = {"round": round_number, "status": "evaluation_failed",
                           "accepted": False, "error": str(exc)[:1000], **proposal}
        state["attempts"].append(attempt)
        state["last_updated_at"] = time.time()
        _save_state(run_dir, state)
        ledger.append({"type": "persistent_round", **_metric_summary(attempt)})
        print(json.dumps(_metric_summary(attempt), sort_keys=True), flush=True)

    state["last_halt_reason"] = halt_reason or "invocation round limit"
    state["last_updated_at"] = time.time()
    _save_state(run_dir, state)
    ledger.append({"type": "persistent_loop_halt", "reason": state["last_halt_reason"],
                   "round": state["round"], "usd": guard.usd, "calls": guard.calls})
    print(json.dumps({"status": "paused", "reason": state["last_halt_reason"],
                      "round": state["round"], "best_selection_score":
                      state.get("best_selection_score"), "usd": guard.usd,
                      "calls": guard.calls}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
