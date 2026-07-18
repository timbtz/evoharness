"""S1 Feedback: score_only | reflections (ReEvo-style verbal gradients, adapted — MIT)."""
from __future__ import annotations

from core.candidate import Candidate, Pool, PromptSection

_ST_TEMPLATE = """Below are two versions of the evolved function. The second performs better.

[Worse code] (score {worse:.4f})
```python
{worse_code}
```

[Better code] (score {better:.4f})
```python
{better_code}
```

Give hints for designing better heuristics, based on these two versions, in less than 20 words."""

_LT_TEMPLATE = """Below is your prior long-term reflection on designing heuristics for this task:
{prior}

Below are newly gained insights:
{new}

Write constructive hints for designing better heuristics, based on prior reflections and new insights, in less than 50 words."""


def _parent_sections(pool: Pool, parents: list[Candidate], last: Candidate | None) -> list[PromptSection]:
    out = [
        PromptSection(
            f"Current program (train score {p.score('train'):.4f})",
            f"```python\n{p.code}\n```",
        )
        for p in parents
    ]
    if last is not None and last.meta.get("error"):
        out.append(PromptSection(
            "Previous attempt failed — avoid this mistake",
            str(last.meta["error"])[:800],
        ))
    return out


class ScoreOnly:
    def __init__(self, task, llm=None, roles=None, temps=None):
        pass

    def build_context(self, pool, parents, last) -> list[PromptSection]:
        return _parent_sections(pool, parents, last)


class Reflections:
    """score_only context + short-term pair hint + accumulated long-term advice."""

    def __init__(self, task, llm, roles, temps):
        self.llm, self.roles, self.temps = llm, roles, temps
        self.long_term = ""

    def _reflect(self, prompt: str) -> str:
        spec = self.roles.feedback_model()
        return self.llm.chat(
            spec, [{"role": "user", "content": prompt}],
            temperature=self.temps["reflect"], role="reflect", max_tokens=1024,
        ).strip()

    def build_context(self, pool, parents, last) -> list[PromptSection]:
        sections = _parent_sections(pool, parents, last)
        live = sorted(
            {c.score("train"): c for c in pool.all if not c.meta.get("error")}.values(),
            key=lambda c: c.score("train"),
        )
        if len(live) >= 2:
            worse, better = live[-2], live[-1]
            st = self._reflect(_ST_TEMPLATE.format(
                worse=worse.score("train"), worse_code=worse.code,
                better=better.score("train"), better_code=better.code,
            ))
            sections.append(PromptSection("Reflection", st))
            self.long_term = self._reflect(_LT_TEMPLATE.format(
                prior=self.long_term or "(none yet)", new=st,
            ))
        if self.long_term:
            sections.append(PromptSection("Accumulated advice", self.long_term))
        return sections
