# Python orchestration and time-budget tuning

Reshuffling the Python driver (slice lengths, polish pass limits, validity-check frequency, skipping overlays for large N, n-gated budget splits) was neutral-to-negative in every evaluated attempt.

## How it was tried
- c0015 (cvrp-s3): continuous small-burst C calls. Slightly worse.
- c0040, c0041, c0051, c0055 (cvrp-s3/s5): reduced LNS slices, n-gated overlays, reduced overlay passes. All score-neutral (exact ties).
- c0004 (cvrp-s11-66566581): single bounded greedy pass with early-exit on stall. Rejected (-0.1529).
- c0005 (cvrp-s11-66566581): budget-fill loop replacing 5 fixed passes. Accepted but functionally identical to tying (score -0.1064).
- c0008r1 (cvrp-s11-66566581): pushed C LNS budget to maximum by shrinking buffer to 0.15s and reducing overlay to single pass. Train -0.126, val -0.515. Rejected.
- c0025 (cvrp-s11-66566581): dynamic n-gated budget split (85% C for n<=150, 97% for large n). Train -0.121, val -0.271. Rejected.
- c0009, c0011, c0013r1, c0015, c0018 (cvrp-s13-71671014): restored single-call C LNS (reverting c0001's static 3-phase split) while tweaking the overlay orchestration (e.g., swapping 5 fixed rounds for a `while time.monotonic() < t_end` budget-fill loop, changing the exact `deadline` buffer). All scored between -0.095 and -0.138 (train ties, noise).
- c0021 (cvrp-s13-71671014): reverted c0001's 3-phase split to single-call but aggressively increased the overlay reserve (`left = deadline - 0.30`). Train -0.956, val -0.914. Massive regression: stealing just ~0.25s of budget from the LNS causes catastrophic time-starvation.
- c0003 (run cvrp-s19-83885116): guaranteed overlay time slice by reducing C LNS deadline margin from 0.35 to 0.30. Train -0.1230, val -0.2746. Rejected.
- c0014 (run cvrp-s19-83885116): consolidated budget for single C call, removed `_py_fast_oropt` tier. Train -0.1516, val -0.2921. Rejected.
- c0015 (run cvrp-s19-83885116): skipped overlay if < 0.7s left, tightened KNN limit to 5. Train -0.1499, val -0.2546. Rejected.

## Why it failed
- Evaluated variants only moved time between components whose marginal value is below eval noise (see performance-analysis/score-noise-and-gate.md); none changed what the search fundamentally does.
- The overlay converges in 1-2 passes; whether you run exactly 5 passes or infinite `while` loops until timeout, the measured behavior is identical.
- Maximizing the C time budget just eats the safety buffer and risks the 27825 starvation attractor without unlocking meaningfully more SA iterations.

## Verdict
exhausted for micro-tuning (slice sizes, pass limits, early exits, n-gating, dynamic budget splits, overlay budget reservations). The marginal value of Python micro-orchestration is zero.
