"""Curated task knowledge, independent of the reviewer's writable run memory.

Resources are versioned JSON documents bundled with each task. Both prompt
injection and bounded tool retrieval use the same page ranking and contents.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from evoharness.engine.candidate import PromptSection


class Off:
    def __init__(self, knowledge_path=None, mode="inject"):
        pass

    def prompt_sections(self, task, query_hint: str) -> list[PromptSection]:
        return []

    def tools(self) -> list[dict]:
        return []


class TaskKnowledge:
    MAX_READS = 2

    def __init__(self, knowledge_path: Path, mode: str = "inject"):
        self.path = Path(knowledge_path)
        self.mode, self.reads = mode, 0
        resource = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.index = resource.get("index", "")
        self.pages = resource.get("pages", {})

    def _top_pages(self, query: str, k: int = 2) -> list[str]:
        terms = set(re.findall(r"[a-z]{4,}", query.lower()))
        def score(name: str) -> float:
            words = re.findall(r"[a-z]{4,}", self.pages[name].lower())
            return sum(word in terms for word in words) / (len(words) + 1)
        return sorted(sorted(self.pages), key=score, reverse=True)[:k]

    def prompt_sections(self, task, query_hint: str) -> list[PromptSection]:
        sections = [PromptSection("Task knowledge index", self.index)] if self.index else []
        if self.mode == "inject":
            sections.extend(PromptSection(f"Task knowledge: {name}", self.pages[name])
                            for name in self._top_pages(query_hint or task.description))
        else:
            self.reads = 0
            sections.append(PromptSection("Knowledge access",
                f"Use read_knowledge(path) for the indexed pages (max {self.MAX_READS} reads)."))
        return sections

    def tools(self) -> list[dict]:
        if self.mode != "tool":
            return []
        return [{"type": "function", "function": {
            "name": "read_knowledge", "description": "Read a curated task knowledge page.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "Page identifier from the index"}},
                "required": ["path"]}}}]

    def call_tool(self, name: str, args: dict) -> str:
        if name != "read_knowledge":
            return f"unknown tool {name}"
        self.reads += 1
        if self.reads > self.MAX_READS:
            return "read limit reached — answer with what you have"
        page = str(args.get("path", ""))
        return self.pages.get(page, f"no such page; index lists: {sorted(self.pages)}")
