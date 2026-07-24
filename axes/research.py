"""S4 Knowledge option 3: web — a research analyst with real internet access.

Design (2026-07-23, user decision): research lives on the REVIEWER side, never in the
writer loop — writers stay fast idea-pickers over the injected memory index. Context is
protected by a deterministic map-reduce instead of one giant agentic session: one call
frames 2-3 cross-domain questions from the wiki's open directions; each question gets a
FRESH-context tool session (web_search + fetch_url); one synthesis call distills the
notes into new-ideas/web-*.md pages that writers pick up through the shared memory wiki
(so feedback=memory is required — config enforces it). Rules: IDEAS, never code (no
GPL derivation); every page cites exact source URLs, giving the provenance chain
Idea line -> page -> URL. Backend is DuckDuckGo HTML: the z.ai web_search resource
pack is not included in the coding-plan subscription (probed 2026-07-23: standalone
endpoint -> error 1113, in-chat tool -> request hangs).
"""
from __future__ import annotations

import html
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from . import memory as _mem
from .knowledge import Off

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"}
_FILE_RE = r"^=== FILE: (\S+) ===\n(.*?)(?=^=== FILE: |\Z)"


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


_ZAI_MCP = "https://api.z.ai/api/mcp/web_search/mcp"


def _zai_search(query: str, count: int = 10) -> list[dict] | None:
    """z.ai web-search MCP (streamable HTTP, coding key). Returns None when
    unavailable — as of 2026-07-23 every tools/call gets 429 "no resource package"
    on the coding plan; this stays the preferred backend so activating the pack
    (or swapping the key) upgrades runs with no code change."""
    key = os.environ.get("ZAI_API_KEY")
    if not key:
        return None

    def post(body: dict, sid: str | None):
        req = urllib.request.Request(_ZAI_MCP, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **({"Mcp-Session-Id": sid} if sid else {})})
        r = urllib.request.urlopen(req, timeout=20)
        m = re.search(r"^data:(.*)$", r.read().decode("utf-8", "replace"), re.M)
        return (json.loads(m.group(1)) if m else None), r.headers.get("Mcp-Session-Id")

    try:
        _, sid = post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "evoharness", "version": "1"}}}, None)
        resp, _ = post({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "webSearchPro",
            "arguments": {"search_query": query[:70], "count": count}}}, sid)
        result = resp["result"]
        if result.get("isError"):
            return None
        rows = json.loads(result["content"][0]["text"])
        if isinstance(rows, dict):
            rows = rows.get("search_result") or []
        return [{"title": _strip(str(r.get("title", ""))), "url": str(r.get("link", r.get("url", ""))),
                 "snippet": _strip(str(r.get("content", "")))[:300]} for r in rows]
    except Exception:
        return None


def web_search(query: str, count: int = 8) -> list[dict]:
    """z.ai MCP when the account has the resource pack, else DuckDuckGo HTML
    (GET returns a bare form — POST is required)."""
    rs = _zai_search(query, count)
    if rs:
        return rs[:count]
    data = urllib.parse.urlencode({"q": query, "b": ""}).encode()
    req = urllib.request.Request("https://html.duckduckgo.com/html/", data=data, headers=_UA)
    page = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
    links = re.findall(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page)
    snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    out = []
    for (u, t), s in zip(links, snips + [""] * len(links)):
        u = urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("uddg", [u])[0]
        out.append({"title": _strip(t), "url": u, "snippet": _strip(s)[:300]})
    return out[:count]


def fetch_url(url: str, cap: int = 8000) -> str:
    if not str(url).startswith(("http://", "https://")):
        return "only http(s) URLs can be fetched"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=20).read(400_000)
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw.decode("utf-8", "replace"))
    return _strip(text)[:cap]


_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the open web. Returns titles, URLs and snippets.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch one web page as plain text (truncated).",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
]

_QUESTIONS = """You are the research analyst of an evolutionary code-optimization system.
The optimizer is stuck inside this problem's local literature; your job is to find
NON-OBVIOUS directions on the open internet that its memory wiki does not contain.

# Task
{task}

# Memory wiki index (what has been tried; note the open directions)
{index}

Write 2-3 focused, searchable research questions, each on its own line as
`QUESTION: <question>`. At least one must look OUTSIDE this problem's home domain
(analogies from other fields, adjacent algorithmic communities, engineering practice)."""

_SESSION_SYS = """You research ONE question using web_search(query) and fetch_url(url).
Search 2-3 distinct phrasings/angles, then fetch the 1-2 most promising pages.
Extract IDEAS and mechanisms, NEVER code — copying source code is forbidden (license
constraints, including GPL). Finish with compact notes: each idea's mechanism, the
evidence for it, and the exact source URL it came from."""

_SYNTH = """Below: the task, the memory wiki index, and research notes from {k} web
research sessions. Distill the strongest 1-2 ideas that are genuinely NEW relative to
the index into wiki pages a code-writing model can act on. Translate each idea into
this task's concrete terms; IDEAS only, never code from sources (no GPL derivation).
Output ONLY page blocks, nothing else:

=== FILE: new-ideas/web-<slug>.md ===
# <title>
<one-line claim>
## Mechanism (translated to this task)
## How to try it here
<concrete code-free steps a writer can implement>
## Prediction
<falsifiable: which instances/metrics should improve and why>
## Sources
- <url> (<venue/date>)

# Task
{task}

# Memory wiki index
{index}

# Research notes
{notes}"""


class Researcher(Off):
    """Fires every EVERY candidates, or on STALL rejects in a row (COOLDOWN apart)."""

    EVERY, STALL, COOLDOWN, MAX_SESSIONS, ROUNDS = 30, 15, 15, 3, 6

    def __init__(self, wiki_dir=None, mode="inject"):
        self.dir: Path | None = None
        self.since = self.since_accept = self.sessions = 0

    def bind(self, task, llm, roles, temps, ledger) -> None:
        self.task, self.llm, self.roles, self.temps, self.ledger = task, llm, roles, temps, ledger
        self.dir = _mem._ROOT / "memory" / task.name  # the shared memory wiki (feedback=memory)

    def observe(self, pool, cand, accepted: bool) -> None:
        if self.dir is None or cand.meta.get("seed"):
            return
        self.since += 1
        self.since_accept = 0 if accepted else self.since_accept + 1
        if self.sessions >= self.MAX_SESSIONS:
            return
        if not (self.since >= self.EVERY
                or (self.since_accept >= self.STALL and self.since >= self.COOLDOWN)):
            return
        self.since, self.sessions = 0, self.sessions + 1
        try:
            self._research()
        except Exception as e:  # network/LLM/budget trouble must never kill the run
            self.ledger.append({"type": "research_error", "error": str(e)[:300]})

    def _research(self) -> None:
        index = (self.dir / "index.md").read_text() if (self.dir / "index.md").exists() else ""
        model, temp = self.roles.feedback_model(), self.temps["reflect"]
        text = self.llm.chat(model, [{"role": "user", "content": _QUESTIONS.format(
            task=self.task.description, index=index)}], temperature=temp, role="researcher")
        questions = [q.strip() for q in re.findall(r"^QUESTION:\s*(.+)$", text, re.M)][:3]
        sources: list[str] = []

        def handler(name: str, args: dict) -> str:
            if name == "web_search":
                rs = web_search(str(args.get("query", "")))
                sources.extend(r["url"] for r in rs[:3])
                return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['snippet']}" for r in rs) \
                    or "no results"
            if name == "fetch_url":
                sources.append(str(args.get("url", "")))
                return fetch_url(str(args.get("url", "")))
            return f"unknown tool {name}"

        notes = []
        for q in questions:
            try:
                notes.append(f"### {q}\n" + self.llm.chat(
                    model, [{"role": "system", "content": _SESSION_SYS},
                            {"role": "user", "content": f"Question: {q}\n\nTask context: "
                                                        f"{self.task.description}"}],
                    temperature=temp, role="researcher",
                    tools=_TOOLS, tool_handler=handler, rounds=self.ROUNDS))
            except Exception as e:
                notes.append(f"### {q}\n(session failed: {str(e)[:120]})")
        text = self.llm.chat(model, [{"role": "user", "content": _SYNTH.format(
            k=len(notes), task=self.task.description, index=index, notes="\n\n".join(notes))}],
            temperature=temp, role="researcher")
        files = []
        idx = self.dir / "index.md"
        for path, content in re.findall(_FILE_RE, text, re.M | re.S):
            rel = Path(path)
            if not (len(rel.parts) == 2 and rel.parts[0] == "new-ideas" and rel.suffix == ".md"
                    and rel.name.startswith("web-") and ".." not in rel.parts):
                continue
            (self.dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.dir / rel).write_text(content.strip() + "\n")
            files.append(str(rel))
            if idx.exists() and rel.name not in idx.read_text():
                claim = next((l for l in content.splitlines()
                              if l.strip() and not l.startswith("#")), "")[:80]
                idx.write_text(idx.read_text().rstrip()
                               + f"\n- {rel} — web-researched, unevaluated: {claim}\n")
        self.ledger.append({"type": "research", "questions": questions, "files": files,
                            "sources": list(dict.fromkeys(sources))[:20]})
