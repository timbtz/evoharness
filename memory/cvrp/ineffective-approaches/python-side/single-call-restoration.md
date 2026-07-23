# Restoring single-call C LNS (from multi-phase parent)

When operating on a parent with an untrusted static multi-phase C LNS split, restoring the single continuous C LNS call was attempted 14 times across cvrp-s13-71671014 and cvrp-s17-78412700. All 14 attempts were rejected or tied.

## How it was tried
- c0002 through c0023 (12 candidates in cvrp-s13-71671014): reverted c0001's static 3-phase split to a single full-budget C call while tweaking Python overlay orchestration. All scored between -0.073 and -0.956 (rejected as ties or regressions). Writers incorrectly predicted private large-n (n=200-303) improvements via uninterrupted SA cooling, but the `train`/`val` instances (n=101-176) converge perfectly well regardless of static vs. adaptive schedules, masking any deltas.
- c0001 (run cvrp-s17-78412700): restored single-call C LNS from c0000 (a static 3-phase parent). Scored train -0.1279 / val -0.2302 (noise-band tie). 
- c0018 (run cvrp-s17-78412700): restored c0035's unified overlay alongside single-call C LNS, removed segment-shift. Train -0.0998, val -0.2770. Tie.

## Why it failed
- Restoring single-call C LNS has zero measurable impact on train/val instances within the eval noise band.
- Attempting to pair the single-call restoration with a new Python overlay mechanism (like a regime dispatcher) doesn't help establish the overlay's value because the diff size increases the chance of generation truncation (as seen with c0001).

## Verdict
exhausted. Do not propose simply "restoring single-call C LNS" from a multi-phase parent. It is a safe baseline but provides no score gradient to pass the gate.
