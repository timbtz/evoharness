# Candidate-fitted surrogates
Within its 72-eval budget a candidate may fit a cheap model (GP / RBF / polynomial on its own fm results) and use it to pick the next boundary to spend a real eval on.

## Status
- Untested. numpy/scipy available in-sandbox; 72 points in ~20-40 D is thin — local surrogates around the incumbent are more plausible than global ones.
- HARD RULE (paper's surrogate self-deception failure mode): surrogate scores NEVER leave the candidate. Only real fm evals reach the log/archive, and only the returned boundary's clean re-eval is scored. A candidate that returns a surrogate-optimal-but-unevaluated boundary gambles its fitness on an untested point — usually a bad idea.

## Verdict
promising for eval-efficiency; keep expectations low until a variant shows a real shaped-score win over plain greedy at equal budget.
