"""In-run analyst (DAG campaign, 2026-07-24; redesigned 2026-07-25; ported off
Claude to z.ai 2026-07-27 per user directive "only z.ai models with the z.ai API
key"): every `every` candidates an independent z.ai session reviews the search —
the recent window (writer + refiner candidates) and the whole memory wiki (inline
index + read_memory/grep_memory tools) — then DECIDES: continue the current
direction, revive a promising abandoned solution, or pivot to something novel
(with web_search/fetch_url when `web`). It proposes exactly ONE new candidate.

Injection is now written HOST-SIDE from the reply's fenced code blocks instead of
by a model file-write tool. In branch B4 every model-side INJECT write was
permission-blocked, so eight analyst sessions produced zero candidates; parsing
the reply cannot fail that way. Its analysis + decision log lands in the wiki as
new-ideas/analyst-*.md + an index line, ledgered as type "analyst". Calls count
against the run's z.ai budget guard (same bill now). A failure never kills the run."""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_PROMPT = """You are an independent ANALYST embedded in an evolutionary \
code-optimization run ({run_id}, {n} candidates so far). You are NOT the writer: \
your job is meta-analysis of the search itself. Known recurring problems for this \
task: feasibility-margin camping, vlf blindness (train-identical changes that are \
untested), eval starvation (12-27s/eval), missing basin diversity.

Use read_memory/grep_memory to dig through the wiki before you decide. Review the \
recent window below plus whatever the wiki tells you, then DECIDE one of: CONTINUE \
the current direction (if the recent candidates look promising), REVIVE an \
interesting abandoned solution from earlier in this run or a previous run, or \
PIVOT — propose something genuinely novel that combines mechanisms this search has \
not tried, including ideas transplanted from other domains. Ground every claim in \
code and scores — no generic advice. Never re-propose what the wiki marks \
refuted/exhausted.

Write a markdown page (no preamble) — it is saved to the wiki as your decision \
record, so future writers and analysts must be able to trace what you decided, \
why, and what you already ruled out. Exactly this skeleton:
# Analyst notes — {run_id} @ {n} candidates
## What the search is doing
## Binding problem(s) now
## Decision: continue | revive | pivot — and why
## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
## Decision log (alternatives considered and rejected, with reasons)

# Task
{description}

# Memory wiki
{wiki}

# Current best ({best_id}, train {best_score})
```python
{best_code}
```

# Recent attempts (newest last)
{recent}"""

_WEB = """\nYou have web_search(query)/fetch_url(url): spend part of the session \
looking OUTSIDE the wiki (papers, competition write-ups, optimizer literature, \
other domains) for mechanisms this search has not tried, and when you pivot, \
ground the proposal in what you find (cite the source URL)."""

_INJECT = """\nINJECTION: turn your decision into code. AFTER the markdown page, \
append exactly {n} fenced ```python block(s) — nothing between or after them. The \
run evaluates + gates them like writer candidates next generation, and the \
following writer candidates optimize on top of whatever is accepted, so submit the \
ONE program you most believe in (continuation, revival, or pivot). Each block must \
be a COMPLETE standalone module with the same contract as the current best program \
shown below (same entry-point signature, all imports/helpers included), and must \
start with exactly two comment lines:\n# Idea: <one sentence>\n\
# Prediction: <falsifiable expected effect and mechanism>"""

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


class Analyst:
    MAX_SESSIONS = 8

    def __init__(self, model: str = "glm-5.2", every: int = 10,
                 temperature: float = 0.7, web: bool = False, inject: int = 0):
        self.web, self.inject = bool(web), max(0, int(inject))
        self.model, self.every = model, max(1, int(every))
        self.temperature = temperature
        self.run_dir: Path | None = None
        self.ledger = None
        self.llm = None
        self.mem: Path | None = None
        self.wiki = None
        self.run_id = ""
        self._last = 0
        self.k = 0

    def bind(self, run_dir: Path, ledger, task, llm=None, wiki=None) -> None:
        self.run_dir, self.ledger, self.llm = Path(run_dir), ledger, llm
        self.run_id = self.run_dir.name
        self.mem = _ROOT / "memory" / task.name
        self.wiki = wiki if hasattr(wiki, "call_tool") else None
        self.task = task

    def _wiki(self) -> str:
        idx = self.mem / "index.md"
        if not idx.exists():
            return "(no wiki)"
        parts = [f"--- index.md ---\n{idx.read_text()[:4000]}"]
        for p in sorted(self.mem.glob("*/**/*.md")) + sorted(self.mem.glob("*/*.md")):
            parts.append(f"--- {p.relative_to(self.mem)} ---\n{p.read_text()[:2500]}")
        return "\n\n".join(dict.fromkeys(parts))[:40000]

    def _tools(self):
        """Wiki read/grep, plus the research axis' web tools when web=True."""
        tools, handlers = [], {}
        if self.wiki is not None:
            tools += self.wiki.tools()
            handlers.update({t["function"]["name"]: self.wiki.call_tool
                             for t in self.wiki.tools()})
        if self.web:
            from axes import research
            tools += research._TOOLS
            handlers["web_search"] = lambda n, a: "\n".join(
                f"{h.get('title','')} — {h.get('url','')}\n{h.get('snippet','')}"
                for h in research.web_search(str(a.get("query", ""))))
            handlers["fetch_url"] = lambda n, a: research.fetch_url(str(a.get("url", "")))
        if not tools:
            return None, None

        def dispatch(name: str, args: dict) -> str:
            fn = handlers.get(name)
            if fn is None:
                return f"unknown tool {name}"
            try:
                return fn(name, args)
            except Exception as e:
                return f"tool error: {str(e)[:200]}"
        return tools, dispatch

    def maybe(self, pool, n_cands: int, recent: list) -> None:
        if self.mem is None or n_cands - self._last < self.every \
                or self.k >= self.MAX_SESSIONS:
            return
        self._last = n_cands
        self.k += 1
        try:
            self._session(pool, n_cands, recent)
        except Exception as e:  # never kill the run
            self.ledger.append({"type": "analyst_error", "model": self.model,
                                "error": str(e)[:300]})

    def _session(self, pool, n_cands: int, recent: list) -> None:
        best = pool.best("train")
        recent_lines = "\n".join(
            f"- {d['id']} [{'ERR' if d.get('error') else ('acc' if d.get('accepted') else 'rej')}]"
            f" train {d.get('score')} val {d.get('val')}: {(d.get('idea') or '')[:140]}"
            for d in (recent or [])) or "(none)"
        prompt = _PROMPT.format(
            run_id=self.run_id, n=n_cands, description=self.task.description,
            wiki=self._wiki(), best_id=best.id if best else "?",
            best_score=round(best.score("train"), 4) if best else "?",
            best_code=(best.code if best else "")[:12000], recent=recent_lines)
        if self.web:
            prompt = prompt.replace("# Task", _WEB + "\n\n# Task", 1)
        if self.inject:
            prompt = prompt.replace("# Task", _INJECT.format(n=self.inject)
                                    + "\n\n# Task", 1)
        tools, dispatch = self._tools()
        text = (self.llm.chat(
            self.model, [{"role": "user", "content": prompt}], self.temperature,
            role="analyst", tools=tools, tool_handler=dispatch, rounds=6) or "").strip()
        if not text:
            raise RuntimeError("empty analyst reply")

        blocks = [b.strip() for b in _FENCE.findall(text)] if self.inject else []
        page = (text[:text.index("```")] if (blocks and "```" in text) else text).strip()
        if not page.lstrip().startswith("#"):
            page = f"# Analyst notes — {self.run_id} @ {n_cands} candidates\n\n" + page
        rel = Path("new-ideas") / f"analyst-{self.run_id}-{self.k}.md"
        (self.mem / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.mem / rel).write_text(page[:12000].rstrip() + "\n")
        idx = self.mem / "index.md"
        line = f"- {rel.name} — in-run analysis + decision log @ {n_cands} cands ({self.run_id})"
        if rel.name not in idx.read_text():
            idx.write_text(idx.read_text().rstrip() + "\n" + line + "\n")

        ev = {"type": "analyst", "file": str(rel), "n_candidates": n_cands}
        if self.inject:
            inj = self.run_dir / "INJECT"
            inj.mkdir(exist_ok=True)
            written = []
            for i, code in enumerate(blocks[: self.inject], 1):
                p = inj / f"{self.k}-{i}-analyst.py"
                p.write_text(code + "\n")
                written.append(p.name)
            ev["inject_files"] = written
            if not written:
                ev["inject_warning"] = "reply contained no fenced code block"
        self.ledger.append(ev)


ClaudeAnalyst = Analyst  # back-compat for saved configs/state referring to the old name
