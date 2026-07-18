"""S4 Knowledge: off | wiki_fs — per-task markdown wiki with index.md (LLM-wiki style).

Mode A "inject" (default): index.md + top-2 pages by keyword overlap with the query hint.
Mode B "tool": index.md only + a read_wiki(path) tool the writer may call (max 2 reads).
Precedent: KernelEvolve's filesystem knowledge base (arXiv 2512.23236).
"""
from __future__ import annotations

import re
from pathlib import Path

from core.candidate import PromptSection


class Off:
    def __init__(self, wiki_dir=None, mode="inject"):
        pass

    def prompt_sections(self, task, query_hint: str) -> list[PromptSection]:
        return []

    def tools(self) -> list[dict]:
        return []


class WikiFS:
    MAX_READS = 2

    def __init__(self, wiki_dir: Path, mode: str = "inject"):
        self.dir = Path(wiki_dir)
        self.mode = mode
        self.reads = 0

    def _pages(self) -> list[Path]:
        return sorted(p for p in self.dir.glob("*.md") if p.name != "index.md")

    def _top_pages(self, query: str, k: int = 2) -> list[Path]:
        terms = set(re.findall(r"[a-z]{4,}", query.lower()))
        def score(page: Path) -> float:
            words = re.findall(r"[a-z]{4,}", page.read_text().lower())
            hits = sum(1 for w in words if w in terms)
            return hits / (len(words) + 1)
        return sorted(self._pages(), key=score, reverse=True)[:k] if terms else self._pages()[:k]

    def prompt_sections(self, task, query_hint: str) -> list[PromptSection]:
        index = self.dir / "index.md"
        sections = [PromptSection("Knowledge wiki index", index.read_text())] if index.exists() else []
        if self.mode == "inject":
            for page in self._top_pages(query_hint or task.description):
                sections.append(PromptSection(f"Wiki: {page.name}", page.read_text()))
        else:
            self.reads = 0
            sections.append(PromptSection(
                "Wiki access",
                f"You may call read_wiki(path) on the pages listed in the index (max {self.MAX_READS} reads).",
            ))
        return sections

    def tools(self) -> list[dict]:
        if self.mode != "tool":
            return []
        return [{
            "type": "function",
            "function": {
                "name": "read_wiki",
                "description": "Read a markdown page from the task knowledge wiki.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "page filename, e.g. heuristics.md"}},
                    "required": ["path"],
                },
            },
        }]

    def call_tool(self, name: str, args: dict) -> str:
        if name != "read_wiki":
            return f"unknown tool {name}"
        self.reads += 1
        if self.reads > self.MAX_READS:
            return "read limit reached — answer with what you have"
        page = (self.dir / Path(str(args.get("path", ""))).name)  # basename only: no traversal
        return page.read_text() if page.exists() else f"no such page; index lists: {[p.name for p in self._pages()]}"
