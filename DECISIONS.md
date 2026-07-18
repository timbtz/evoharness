# Deviations from the briefing (one line each)

- Cheap role model = `glm-4.5-air` ($0.2/$1.1 per 1M): cheapest z.ai model that reliably emits fenced code; strong = `glm-5.2`.
- Wiki knowledge: Mode A "inject" (index + top-2 pages by keyword overlap) is the default; Mode B "tool" (read_wiki, ≤2 reads) is implemented and selected via `Config.wiki_mode`.
- `Task` grew a `description` attribute (the prompt header) on top of the §3 protocol — every task needs it, the loop stays generic.
- Objective scalars: `quality_per_dollar` = (score − seed score)/max(usd, 0.01) at gate time; `time_capped` ≡ `quality` because the budget guard already enforces `max_seconds` for every run.
- Candidate ids are deterministic (`c0000` seed, `c{gen:04d}`, migrants `{id}m{t}`) so golden test 6 holds across processes.
- Binpacking private split: Zenodo 14162744 was inspected but rejected (capacity-150 / 250-item protocol mismatches our capacity-100 / 5000-item setup) → generated OOD instances instead (2× Weibull(1.5)·30, 2× uniform 20..70).
- PyVRP added as a ninth reference repo (user request): its Statistics/Result split maps to our ledger JSONL (write side) + `experiments/report.py` (read side, incl. CSV export à la `Statistics.to_csv`).
- GLM thinking mode disabled on every call (`thinking: {type: disabled}`): with it on, glm-5.2 spent 2.6k–16k+ completion tokens per call and sometimes never emitted content before the cap (parse errors at $0.07/call); with it off, the same prompt yields a clean code block in ~240 tokens. Evolution wants many cheap samples, not per-call deliberation.
- `parse_code` accepts a final unterminated ```python block (max_tokens truncation) — a syntactically broken tail just scores −inf in the sandbox; writer `max_tokens` raised 4096 → 16384.
- Docker: single `api` service (runs execute as threads inside the API process — a separate worker service would duplicate state for no isolation gain at this scale); inside containers without user namespaces the sandbox auto-falls back from `unshare -rn` to the socket-stub preamble.
