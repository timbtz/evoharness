"""In-run Opus analyst (DAG campaign, 2026-07-24; redesigned 2026-07-25): every
`every` candidates an independent headless Claude session (`claude -p`) reviews
the search — the recent window (writer + refiner candidates), the wiki, and as
much on-disk run history as it wants (Read/Glob/Grep on the ledgers) — then
DECIDES: continue the current direction, revive a promising abandoned solution
from an earlier run, or pivot to something novel (web research allowed, take
your time). It proposes exactly ONE new candidate, written to <run_dir>/INJECT/
(inject=1) — the loop evaluates and gates it like a writer candidate next
generation, and subsequent writer candidates optimize on top of whatever is
accepted. Its full analysis + decision log lands in the wiki as
new-ideas/analyst-*.md (namespace-enforced) + an index line, and is ledgered as
type "analyst". A failure never kills the run."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_PROMPT = """You are an independent ANALYST embedded in an evolutionary \
code-optimization run ({run_id}, {n} candidates so far). You are NOT the writer: \
your job is meta-analysis of the search itself. Known recurring problems for this \
task: feasibility-margin camping, vlf blindness (train-identical changes that are \
untested), eval starvation (12-27s/eval), missing basin diversity.

You have Read/Glob/Grep and time — use them. The full history is on disk: this \
run's ledger.jsonl is in your working directory (candidate events carry \
meta.idea, meta.prediction, meta.reasoning, scores, code); earlier runs: \
{runs_dir}/*/ledger.jsonl; the memory wiki: {mem_dir}. Review the recent window \
below plus whatever history you need, then DECIDE one of: CONTINUE the current \
direction (if the recent candidates look promising), REVIVE an interesting \
abandoned solution from earlier in this run or a previous run, or PIVOT — \
propose something genuinely novel that combines mechanisms this search has not \
tried, including ideas transplanted from other domains. Ground every claim in \
code and scores — no generic advice. Never re-propose what the wiki marks \
refuted/exhausted.

Output ONLY a markdown page (no preamble) — it is saved to the wiki as your \
decision record, so future writers and analysts must be able to trace what you \
decided, why, and what you already ruled out. Exactly this skeleton:
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

_WEB = """\nYou have WebSearch/WebFetch: spend part of the session looking OUTSIDE \
the wiki (papers, competition write-ups, optimizer literature, other domains) for \
mechanisms this search has not tried, and when you pivot, ground the proposal in \
what you find (cite the source URL)."""

_INJECT = """\nINJECTION: turn your decision into code — CREATE exactly {n} \
candidate optimizer module(s), written to INJECT/<k>-<slug>.py (relative path). \
The run evaluates + gates them like writer candidates next generation, and the \
following writer candidates optimize on top of whatever is accepted — so submit \
the ONE program you most believe in (continuation, revival, or pivot). Each file \
must be a COMPLETE standalone module with the same contract as the current best \
program shown below (same entry-point signature, all imports/helpers included), \
and must start with exactly two comment lines:\n# Idea: <one sentence>\n\
# Prediction: <falsifiable expected effect and mechanism>"""


class ClaudeAnalyst:
    MAX_SESSIONS = 8

    def __init__(self, model: str = "claude-opus-4-8", every: int = 10,
                 timeout: float = 600.0, web: bool = False, inject: int = 0):
        self.web, self.inject = bool(web), max(0, int(inject))
        if web or inject:
            timeout = max(timeout, 1800.0)  # research + injection sessions take time
        self.model, self.every, self.timeout = model, max(1, int(every)), timeout
        self.run_dir: Path | None = None
        self.ledger = None
        self.mem: Path | None = None
        self.run_id = ""
        self._last = 0
        self.k = 0

    def bind(self, run_dir: Path, ledger, task) -> None:
        self.run_dir, self.ledger = Path(run_dir), ledger
        self.run_id = self.run_dir.name
        self.mem = _ROOT / "memory" / task.name
        self.task = task

    def _wiki(self) -> str:
        idx = self.mem / "index.md"
        if not idx.exists():
            return "(no wiki)"
        parts = [f"--- index.md ---\n{idx.read_text()[:4000]}"]
        for p in sorted(self.mem.glob("*/**/*.md")) + sorted(self.mem.glob("*/*.md")):
            parts.append(f"--- {p.relative_to(self.mem)} ---\n{p.read_text()[:2500]}")
        return "\n\n".join(dict.fromkeys(parts))[:40000]

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
            best_code=(best.code if best else "")[:12000], recent=recent_lines,
            runs_dir=str(_ROOT / "runs"), mem_dir=str(self.mem))
        allowed = ["Read", "Glob", "Grep"]  # run-history + wiki review
        if self.web:
            prompt = prompt.replace("# Task", _WEB + "\n\n# Task", 1)
            allowed += ["WebSearch", "WebFetch"]
        if self.inject:
            prompt = prompt.replace("# Task", _INJECT.format(n=self.inject)
                                    + "\n\n# Task", 1)
            (self.run_dir / "INJECT").mkdir(exist_ok=True)
            allowed.append("Write(INJECT/**)")
        t0 = time.monotonic()
        r = subprocess.run(
            ["claude", "-p", "--model", self.model, "--output-format", "text",
             "--strict-mcp-config", "--allowedTools", ",".join(allowed)],
            input=prompt, capture_output=True, text=True,
            timeout=self.timeout, cwd=str(self.run_dir))
        text = (r.stdout or "").strip()
        if r.returncode != 0 and not text:
            raise RuntimeError(f"claude rc={r.returncode}: {r.stderr[-300:]}")
        self.ledger.append({
            "type": "llm_call", "role": "analyst", "model": self.model,
            "usd": None, "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(text) // 4,
            "seconds": round(time.monotonic() - t0, 2), "text": text})
        if not text.lstrip().startswith("#"):
            text = f"# Analyst notes — {self.run_id} @ {n_cands} candidates\n\n" + text
        rel = Path("new-ideas") / f"analyst-{self.run_id}-{self.k}.md"
        (self.mem / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.mem / rel).write_text(text[:12000].rstrip() + "\n")
        idx = self.mem / "index.md"
        line = f"- {rel.name} — in-run Opus analysis + decision log @ {n_cands} cands ({self.run_id})"
        if rel.name not in idx.read_text():
            idx.write_text(idx.read_text().rstrip() + "\n" + line + "\n")
        ev = {"type": "analyst", "file": str(rel), "n_candidates": n_cands}
        if self.inject:
            ev["inject_files"] = sorted(
                p.name for p in (self.run_dir / "INJECT").glob("*.py"))
        self.ledger.append(ev)
