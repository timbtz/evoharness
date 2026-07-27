# The LF-Selection Overwrite Bug
Using an LF-verification phase to reject train-fidelity "landmines" accidentally overrides the train-score winner, returning a lower-scoring candidate and causing silent regressions.
## How it was tried
- `stellar_p2-s105-72881323` c0050f (train 0.6240): Replaced blind train-score selection with an LF landmine gate. The loop selected the first LF-feasible candidate based on novelty-distance rank rather than the highest train `shaped_score`, dropping the score.
- `stellar_p2-s105-72881323` c0051 (train 0.6241): Inherited the buggy loop and regressed.
- `stellar_p2-s105-72881323` c0052 (train 0.6250): Fixed by making LF verification a margin-strengthening tie-breaker only, securely returning the highest train-key candidate.
## Why it failed
The LF loop was intended as a strict gate to drop physics-collapse candidates, but if it iterated over a list sorted by a novelty-penalized key, it naturally favored high novelty over raw train score.
## Verdict
recurring pitfall — Always sort and select by the primary train objective first. If verifying via LF, use it only to tie-break candidates within a tiny delta (e.g., 0.0005) of the top train score, never as an override.
