"""Fable refiner (user experiment 2026-07-23): after each writer candidate is
evaluated, a Claude Fable 5 HEADLESS session (`claude -p`, no tools, prompt-only)
sees that candidate + its eval feedback, the memory wiki, and the recent attempt
history, and proposes a refined v2 that gets evaluated and gated like any other
candidate (id suffix "f"). Fable calls are ledgered as role "refiner" but not
charged to the z.ai budget guard (different bill); the generations cap bounds
them. A refiner failure never kills the run."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from core.candidate import Candidate
from core.llm import parse_code, parse_idea, parse_prediction, parse_reasoning

_PROMPT = """You are the REFINER in an evolutionary code-optimization loop. A \
writer model just produced the candidate below; it has been evaluated. Your job: \
propose a BETTER v2 of it — fix what is broken, keep what works, apply one or two \
well-reasoned improvements grounded in the wiki evidence. Prior evidence says \
minimal surgical changes to the incumbent outperform full restructures here.

Output contract (exactly): one line `Idea: <one-sentence summary>`, one line \
`Prediction: <expected effect and mechanism, falsifiable>`, brief reasoning, then \
exactly ONE fenced code block with the COMPLETE improved module.

# Task
{description}

# Memory wiki
{wiki}

# Incumbent parent (score {parent_score})
```python
{parent_code}
```

# Recent attempts (newest last)
{recent}

# The candidate to refine (id {cand_id}, {verdict})
Writer's idea: {idea}
Scores: {scores}
Eval metrics: {metrics}
Error: {error}
```python
{cand_code}
```"""


class ClaudeRefiner:
    def __init__(self, model: str = "claude-fable-5", timeout: float = 900.0):
        self.model, self.timeout = model, timeout
        self.run_dir: Path | None = None
        self.ledger = None

    def bind(self, run_dir: Path, ledger) -> None:
        self.run_dir, self.ledger = Path(run_dir), ledger

    def _wiki(self, wiki_dir: Path) -> str:
        d = Path(wiki_dir)
        if not (d / "index.md").exists():
            return "(no wiki)"
        parts = [f"--- index.md ---\n{(d / 'index.md').read_text()[:4000]}"]
        pages = sorted(p for p in d.glob("*/**/*.md")) + sorted(d.glob("*/*.md"))
        for p in dict.fromkeys(pages):  # ordered dedupe
            parts.append(f"--- {p.relative_to(d)} ---\n{p.read_text()[:3500]}")
        return "\n\n".join(parts)[:60000]

    def refine(self, task, pool, parent: Candidate, cand: Candidate,
               gen: int, recent: list) -> Candidate | None:
        mem_dir = Path(__file__).resolve().parent.parent / "memory" / task.name
        recent_lines = "\n".join(
            f"- {d['id']} [{'ERR' if d.get('error') else ('acc' if d.get('accepted') else 'rej')}]"
            f" train {d.get('score')}: {(d.get('idea') or '')[:120]}"
            for d in (recent or [])) or "(none)"
        metrics = {k: v for k, v in (cand.meta.get("metrics") or {}).items()
                   if k != "boundary"}
        prompt = _PROMPT.format(
            description=task.description, wiki=self._wiki(mem_dir),
            parent_score=round(parent.score("train"), 4),
            parent_code=parent.code[:20000],
            recent=recent_lines, cand_id=cand.id,
            verdict="accepted" if cand.id in pool.by_id else "rejected",
            idea=cand.meta.get("idea") or "(none)",
            scores={s: round(v, 4) for s, v in cand.scores.items()
                    if v == v and v > float("-inf")},
            metrics=str(metrics)[:2000],
            error=(cand.meta.get("error") or "(none)")[:600],
            cand_code=(cand.code or "(no code)")[:20000])
        t0 = time.monotonic()
        try:
            r = subprocess.run(
                ["claude", "-p", "--model", self.model, "--output-format", "text",
                 "--strict-mcp-config", "--disallowedTools", "*"],
                input=prompt, capture_output=True, text=True,
                timeout=self.timeout, cwd=str(self.run_dir))
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
            code=code, parent_id=parent.id, id=f"c{gen:04d}f",
            meta={"gen": gen, "refiner": self.model, "refined_from": cand.id,
                  "idea": parse_idea(text), "prediction": parse_prediction(text),
                  "reasoning": parse_reasoning(text)})
