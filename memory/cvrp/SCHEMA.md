# How to maintain this memory wiki (you are the reviewer agent)
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
Never delete index.md or this file.