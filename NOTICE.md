# Attribution for adapted code and designs (§9)

Reference clones live in `reference/` (gitignored); see `reference/NOTES.md` for the
per-module "what we take / from where / what we simplify" log.

- **FunSearch** (google-deepmind/funsearch, Apache-2.0): bin-packing task design —
  `priority(item, bins)` convention, argmax packing loop, Weibull instance recipe,
  L1-bound excess metric (`tasks/binpacking/task.py`); islands evolution model,
  compressed to 4 flat islands with ring migration (`axes/search.py`).
- **ReEvo** (ai4co/reevo, MIT): short-/long-term reflection prompt templates
  (`axes/feedback.py`); subprocess-per-eval harness pattern (`tasks/common.py`).
- **EoH** (FeiLiu36/EoH, MIT): population management — keep-first-per-unique-score,
  rank-based parent weights 1/(rank+1+len(pop)) (`axes/search.py`).
- **OpenEvolve** (codelion/openevolve, Apache-2.0): circle-packing evaluator
  (containment/overlap validation, sum-of-radii score) adapted in
  `tasks/circles/task.py`; wiki facts in `tasks/circles/wiki/`.
- **PyVRP** (PyVRP/PyVRP, MIT): tracking design studied — per-iteration Statistics +
  Result + CSV export informed the ledger/report split (`experiments/report.py`).
  No code copied.
- **ShinkaEvolve, LLaMEA, llm4ad**: studied for design comparison; no code copied.
- **bpo_challenge_gdrl**: NO LICENSE — interface-compatibility study only
  (`Planner.plan()` signature noted in `reference/NOTES.md`); no code copied.
- **TSPLIB instances** (via github.com/mastqe/tsplib mirror): classic academic
  benchmark data (Reinelt, 1991); downloaded at setup by `tasks/tsp/fetch.py`,
  not redistributed in this repo.

No GPL/AGPL code was used.
