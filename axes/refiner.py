"""Refiner (redesigned 2026-07-25; ported off Claude to z.ai 2026-07-27, user
directive "only z.ai models with the z.ai API key"): every `refiner_every` writer
candidates ONE z.ai session sees that whole window — each candidate's idea,
prediction, reasoning, eval feedback and code — and synthesizes ONE best version:
refine the most promising candidate or combine the strongest mechanisms of
several, aiming to one-shot a new top candidate with the wiki's evidence. It gets
the wiki index inline plus read_memory/grep_memory tools (the z.ai analogue of the
old Read/Glob/Grep session). The v2 is evaluated and gated like any other
candidate (id suffix "f"). Calls are ledgered as role "refiner" and — unlike the
Claude era — DO count against the run's z.ai budget guard, because they are now
the same bill. A refiner failure never kills the run."""
from __future__ import annotations

from pathlib import Path

from core.candidate import Candidate
from core.llm import parse_code, parse_idea, parse_prediction, parse_reasoning

_PROMPT = """You are the REFINER in an evolutionary code-optimization loop. The \
last {n} writer candidates are shown below in full: idea, prediction, reasoning, \
evaluation feedback, code. They are your raw material. Your job: produce exactly \
ONE candidate that is the best version of this window — refine the single most \
promising candidate, or combine the strongest mechanisms of several — so it has a \
real chance to one-shot a new top candidate. Use read_memory/grep_memory on the \
task's memory wiki before committing to a mechanism: build on what is proven, mine \
new-ideas/, and never re-apply what is marked refuted/exhausted. Prior evidence \
says minimal surgical changes to a strong incumbent outperform full restructures \
here.

Output contract (exactly): one line `Idea: <one-sentence summary>`, one line \
`Prediction: <expected effect and mechanism, falsifiable>`, brief reasoning \
(including WHICH window candidates you drew on and why), then exactly ONE fenced \
code block with the COMPLETE improved module.

# Task
{description}

# Wiki index (read pages with read_memory/grep_memory for details)
{index}

# Current best in pool ({best_id}, train {best_score})
```python
{best_code}
```

# The last {n} candidates (oldest first)
{window}"""


class Refiner:
    def __init__(self, model: str = "glm-5.2", temperature: float = 0.6):
        self.model, self.temperature = model, temperature
        self.run_dir: Path | None = None
        self.ledger = None
        self.llm = None
        self.wiki = None

    def bind(self, run_dir: Path, ledger, llm, wiki=None) -> None:
        self.run_dir, self.ledger, self.llm = Path(run_dir), ledger, llm
        # the memory axis, when feedback=memory: lends its read/grep tools
        self.wiki = wiki if hasattr(wiki, "call_tool") else None

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
        try:
            text = self.llm.chat(
                self.model, [{"role": "user", "content": prompt}],
                self.temperature, role="refiner",
                tools=self.wiki.tools() if self.wiki else None,
                tool_handler=self.wiki.call_tool if self.wiki else None)
        except Exception as e:  # budget guard, 429, network — never kill the run
            self.ledger.append({"type": "llm_error", "role": "refiner",
                                "model": self.model, "error": str(e)[:300]})
            return None
        code = parse_code(text)
        if not code:
            return None
        return Candidate(
            code=code, parent_id=best.id, id=f"c{gen:04d}f",
            meta={"gen": gen, "refiner": self.model,
                  "refined_from": [c.id for c in window],
                  "idea": parse_idea(text), "prediction": parse_prediction(text),
                  "reasoning": parse_reasoning(text)})


ClaudeRefiner = Refiner  # back-compat for saved configs/state referring to the old name
