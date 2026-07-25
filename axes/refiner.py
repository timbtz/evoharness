"""Opus refiner (redesigned 2026-07-25, user spec): every `refiner_every` writer
candidates (default 5) a HEADLESS Claude session (`claude -p`, read-only search
tools, cwd = the task's memory wiki) sees that whole window — each candidate's
idea, prediction, reasoning, eval feedback and code — and synthesizes ONE best
version: refine the most promising candidate or combine the strongest mechanisms
of several, aiming to one-shot a new top candidate with the wiki's evidence. No
web access. The v2 is evaluated and gated like any other candidate (id suffix
"f"). Calls are ledgered as role "refiner" but not charged to the z.ai budget
guard (different bill); the window cadence bounds them. A refiner failure never
kills the run."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from core.candidate import Candidate
from core.llm import parse_code, parse_idea, parse_prediction, parse_reasoning

_PROMPT = """You are the REFINER in an evolutionary code-optimization loop. The \
last {n} writer candidates are shown below in full: idea, prediction, reasoning, \
evaluation feedback, code. They are your raw material. Your job: produce exactly \
ONE candidate that is the best version of this window — refine the single most \
promising candidate, or combine the strongest mechanisms of several — so it has a \
real chance to one-shot a new top candidate. You are running inside the task's \
memory wiki directory with Read/Glob/Grep: search it before committing to a \
mechanism — build on what is proven, mine new-ideas/, and never re-apply what is \
marked refuted/exhausted. Prior evidence says minimal surgical changes to a strong \
incumbent outperform full restructures here.

Output contract (exactly): one line `Idea: <one-sentence summary>`, one line \
`Prediction: <expected effect and mechanism, falsifiable>`, brief reasoning \
(including WHICH window candidates you drew on and why), then exactly ONE fenced \
code block with the COMPLETE improved module.

# Task
{description}

# Wiki index (search pages with Grep/Read for details)
{index}

# Current best in pool ({best_id}, train {best_score})
```python
{best_code}
```

# The last {n} candidates (oldest first)
{window}"""


class ClaudeRefiner:
    def __init__(self, model: str = "claude-opus-4-8", timeout: float = 1200.0):
        self.model, self.timeout = model, timeout
        self.run_dir: Path | None = None
        self.ledger = None

    def bind(self, run_dir: Path, ledger) -> None:
        self.run_dir, self.ledger = Path(run_dir), ledger

    @staticmethod
    def _fmt(c: Candidate, pool) -> str:
        metrics = {k: v for k, v in (c.meta.get("metrics") or {}).items()
                   if k != "boundary"}
        scores = {s: round(v, 4) for s, v in c.scores.items()
                  if v == v and v > float("-inf")}
        return (
            f"## {c.id} — {'ACCEPTED' if c.id in pool.by_id else 'rejected'}, "
            f"scores {scores or '(none)'}\n"
            f"Idea: {c.meta.get('idea') or '(none)'}\n"
            f"Prediction: {c.meta.get('prediction') or '(none)'}\n"
            f"Eval metrics: {str(metrics)[:1500]}\n"
            f"Error: {(str(c.meta.get('error')) if c.meta.get('error') else '(none)')[:600]}\n"
            f"Reasoning: {(c.meta.get('reasoning') or '(none)')[:1500]}\n"
            f"```python\n{(c.code or '(no code)')[:9000]}\n```"
        )

    def refine(self, task, pool, window: list[Candidate], gen: int) -> Candidate | None:
        mem_dir = Path(__file__).resolve().parent.parent / "memory" / task.name
        best = pool.best("train")
        index = (mem_dir / "index.md").read_text()[:5000] \
            if (mem_dir / "index.md").exists() else "(no wiki)"
        prompt = _PROMPT.format(
            n=len(window), description=task.description, index=index,
            best_id=best.id, best_score=round(best.score("train"), 4),
            best_code=best.code[:20000],
            window="\n\n".join(self._fmt(c, pool) for c in window))
        cwd = mem_dir if mem_dir.is_dir() else self.run_dir
        t0 = time.monotonic()
        try:
            r = subprocess.run(
                ["claude", "-p", "--model", self.model, "--output-format", "text",
                 "--strict-mcp-config", "--allowedTools", "Read,Glob,Grep"],
                input=prompt, capture_output=True, text=True,
                timeout=self.timeout, cwd=str(cwd))
            text = r.stdout or ""
            if r.returncode != 0 and not text.strip():
                raise RuntimeError(f"claude rc={r.returncode}: {r.stderr[-300:]}")
        except Exception as e:
            self.ledger.append({"type": "llm_error", "role": "refiner",
                                "model": self.model, "error": str(e)[:300]})
            return None
        self.ledger.append({
            "type": "llm_call", "role": "refiner", "model": self.model,
            "usd": None, "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(text) // 4,
            "seconds": round(time.monotonic() - t0, 2), "text": text})
        code = parse_code(text)
        if not code:
            return None
        return Candidate(
            code=code, parent_id=best.id, id=f"c{gen:04d}f",
            meta={"gen": gen, "refiner": self.model,
                  "refined_from": [c.id for c in window],
                  "idea": parse_idea(text), "prediction": parse_prediction(text),
                  "reasoning": parse_reasoning(text)})
