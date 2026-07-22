"""S1 Feedback option 3: memory — a Karpathy-LLM-wiki run memory kept by a review agent.

Writers only READ the wiki: index.md is injected every generation and read_memory /
grep_memory tools let them query pages. They never write. A separate reviewer (the
feedback model) maintains the wiki from the ledger's reasoning trail — triggered every
REVIEW_EVERY candidates or when a new global best lands, covering only the stretch not
yet reviewed (cap PENDING_CAP). Sections follow ShinkaEvolve's meta-scratchpad; the
index.md-as-library + schema file follow Karpathy's llm-wiki and KernelEvolve
(arXiv 2512.23236). ONE wiki per task at <repo>/memory/<task>/ — every run reads and
extends the same memory, so no run starts from zero; per-run traceability comes from
the ledger (memory_review events + the reviewer's full llm_call text).
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

from core.candidate import Pool, PromptSection

from .feedback import _parent_sections

_ROOT = Path(__file__).resolve().parent.parent

SECTIONS = ("successful-patterns", "ineffective-approaches",
            "implementation-insights", "performance-analysis", "new-ideas")

SCHEMA = """# How to maintain this memory wiki (you are the reviewer agent)
One page per DISTINCT approach/idea family, kebab-case filename, in the right section:
successful-patterns/ (what worked, with evidence), ineffective-approaches/ (what failed
or plateaued — so writers stop retrying it), implementation-insights/ (recurring bugs,
language/runtime pitfalls, fixes), performance-analysis/ (where the score lives:
instance sizes, time budget, noise, bottlenecks), new-ideas/ (untested or barely-tried
ideas worth trying: proposed-but-never-evaluated candidates, promising directions —
good ideas must never get lost).

COVERAGE RULE: every distinct approach tried in the reviewed stretch must appear
exactly once somewhere — its own page or a labeled bullet on an existing family page.
Never drop an approach silently; merge only true duplicates. An idea that was proposed
but never evaluated (parse/compile death) goes to new-ideas/, not oblivion.

Page format — real markdown, information-dense, no narrative filler:
# <title>
<one-line claim>
## How it was tried
- one bullet per variant: candidate ids, what the code ACTUALLY did, score/outcome
## Why it worked / failed
<grounded in the code and results, not the writer's claims — check the code; when the
code contradicts the stated idea, record what the code really did>
## Verdict
promising | exhausted | refuted — and what a future writer should do next.

index.md is the library: one line per page per section ("- file.md — what it holds"),
plus Current best (id, scores, what the program actually is) and Open directions.
Keep facts exact (candidate ids, scores, run names). Beware eval noise: small score
deltas are often noise — never build a claim on a single small delta.
Rewrite pages when merging; keep each under ~80 lines.
Never delete index.md or this file."""

_REVIEW = """You maintain the memory wiki of an evolutionary code-optimization run.
Below: the wiki schema, the current index, existing pages, and the {n} candidate
attempts since the last review (id, accepted?, score, error, the writer's stated idea
and reasoning, and the candidate's CODE — full, or a unified diff vs its parent).
Judge each attempt by its code, not by the writer's self-description. Update the wiki
so future writers avoid repeats and build on what works; apply the COVERAGE RULE.

Output ONLY updated or new files, each as:
=== FILE: <section>/<name>.md ===
<full file content>

Always end with the rewritten index:
=== FILE: index.md ===
<full index content>"""


class MemoryWiki:
    REVIEW_EVERY, PENDING_CAP = 20, 50
    MAX_PAGE, MAX_GREP = 6000, 40

    def __init__(self, task, llm, roles, temps):
        self.task, self.llm, self.roles, self.temps = task, llm, roles, temps
        self.dir: Path | None = None
        self.pending: list[dict] = []
        self.best: float | None = None  # None until the seed sets the baseline

    def bind(self, run_dir: Path, ledger) -> None:
        self.dir = _ROOT / "memory" / self.task.name  # one shared wiki per task
        for s in SECTIONS:
            (self.dir / s).mkdir(parents=True, exist_ok=True)
        if not (self.dir / "SCHEMA.md").exists():
            (self.dir / "SCHEMA.md").write_text(SCHEMA)
        if not (self.dir / "index.md").exists():
            (self.dir / "index.md").write_text(
                "# Memory index\n(empty — first review pending)\n")

    def _pages(self) -> list[Path]:
        return sorted(p for s in SECTIONS for p in (self.dir / s).glob("*.md"))

    def _digest(self, c, accepted: bool, pool: Pool) -> dict:
        code, parent = c.code or "", pool.by_id.get(c.parent_id or "")
        if parent and parent.code and len(code) > 4000:  # long: show the change, not the whole
            code = "".join(difflib.unified_diff(
                parent.code.splitlines(True), code.splitlines(True),
                f"parent {parent.id}", c.id, n=2))
        return {
            "id": c.id, "gen": c.meta.get("gen"), "accepted": accepted,
            "score": c.score("train"), "error": (c.meta.get("error") or "")[:300],
            "novelty": c.meta.get("novelty"), "idea": c.meta.get("idea"),
            "reasoning": (c.meta.get("reasoning") or "")[:2000], "code": code[:6000],
        }

    def build_context(self, pool: Pool, parents, last) -> list[PromptSection]:
        sections = _parent_sections(pool, parents, last)
        if self.dir is None:
            return sections  # unbound (unit use) — plain score_only behaviour
        gb = max((c.score("train") for c in pool.all), default=float("-inf"))
        if last is not None and not last.meta.get("seed"):
            self.pending.append(self._digest(last, last.id in pool.by_id, pool))
            self.pending = self.pending[-self.PENDING_CAP:]
        new_best = self.best is not None and gb > self.best
        if self.pending and (len(self.pending) >= self.REVIEW_EVERY or new_best):
            self._review(pool)
        self.best = gb if self.best is None else max(self.best, gb)
        sections.append(PromptSection(
            "Memory wiki index", (self.dir / "index.md").read_text()))
        sections.append(PromptSection(
            "Memory access",
            "Before choosing your idea, consult the run memory: grep_memory(query) "
            "searches all pages, read_memory(path) reads one (paths as in the index, "
            "e.g. ineffective-approaches/foo.md). Do not repeat approaches recorded "
            "as ineffective; build on successful patterns."))
        return sections

    def _review(self, pool: Pool) -> None:
        existing = "\n\n".join(
            f"--- {p.relative_to(self.dir)} ---\n{p.read_text()[:self.MAX_PAGE]}"
            for p in self._pages()) or "(no pages yet)"
        attempts = "\n".join(str(d) for d in self.pending)
        best = pool.best("train")
        prompt = "\n\n".join([
            _REVIEW.format(n=len(self.pending)), SCHEMA,
            f"## Current index\n{(self.dir / 'index.md').read_text()}",
            f"## Existing pages\n{existing}",
            f"## Current best: {best.id} train {best.score('train'):.4f}",
            f"## Attempts since last review\n{attempts}",
        ])
        text = self.llm.chat(
            self.roles.feedback_model(), [{"role": "user", "content": prompt}],
            temperature=self.temps["reflect"], role="reviewer", max_tokens=32768)
        written = []
        for path, content in re.findall(
                r"^=== FILE: (\S+) ===\n(.*?)(?=^=== FILE: |\Z)", text, re.M | re.S):
            rel = Path(path)
            ok = str(rel) == "index.md" or (
                len(rel.parts) == 2 and rel.parts[0] in SECTIONS
                and rel.suffix == ".md" and ".." not in rel.parts)
            if ok:
                (self.dir / rel).write_text(content.strip() + "\n")
                written.append(str(rel))
        self.llm.ledger.append({"type": "memory_review",
                                "n_candidates": len(self.pending), "files": written})
        self.pending = []

    def tools(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": "read_memory",
                "description": "Read one page of the run memory wiki.",
                "parameters": {"type": "object", "properties": {"path": {
                    "type": "string",
                    "description": "page path from the index, e.g. successful-patterns/foo.md"}},
                    "required": ["path"]}}},
            {"type": "function", "function": {
                "name": "grep_memory",
                "description": "Case-insensitive search across all memory wiki pages.",
                "parameters": {"type": "object", "properties": {"query": {
                    "type": "string", "description": "substring to search for"}},
                    "required": ["query"]}}},
        ]

    def call_tool(self, name: str, args: dict) -> str:
        if name == "read_memory":
            rel = Path(str(args.get("path", "")))
            page = self.dir / Path(*[p for p in rel.parts if p not in ("..", "")])
            return page.read_text()[:self.MAX_PAGE] if page.is_file() \
                else f"no such page; index:\n{(self.dir / 'index.md').read_text()}"
        if name == "grep_memory":
            q = str(args.get("query", "")).lower()
            hits = [f"{p.relative_to(self.dir)}: {ln.strip()}"
                    for p in [*self._pages(), self.dir / "index.md"] if p.exists()
                    for ln in p.read_text().splitlines() if q and q in ln.lower()]
            return "\n".join(hits[:self.MAX_GREP]) or "no matches"
        return f"unknown tool {name}"
