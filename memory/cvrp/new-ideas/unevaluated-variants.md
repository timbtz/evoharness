# Proposed-but-never-scored variants (silent deaths)

Small, mechanism-backed ideas that died before producing a score (no output or parse_error); they carry zero evidence for or against and should not be treated as refuted.

## How it was tried
- c0003 (cvrp-s3-45465035): claimed SA acceptance defect (`cW <= cS` always true). C edit silently failed.
- c0008 (cvrp-s3-45465035): generalize Or-opt in `try_moves` to length 4. Silently failed.
- c0010 (cvrp-s3-45465035): demand-urgency tie-break in recreate + ruin/cooling tuned for n=200-303. Never scored.
- c0001r2 (cvrp-s11-66566581): parse_error death. Code was completely empty.
- c0016r1 (cvrp-s11-66566581): parse_error death. Attempted Or-opt(4, dual-anchor) in Python overlay. Truncated C code led to unterminated string literal.
- c0017 (cvrp-s11-66566581): parse_error death. Attempted to widen C K parameter to 25 for n>=200 instances. Truncated C code. (Note: Successfully evaluated in cvrp-s13-71671014 as c0003/c0018, failed to improve).
- c0018 (cvrp-s11-66566581): empty code/no output. Attempted Python inter-route single-segment relocate to non-KNN positions.
- c0006, c0007, c0010 (cvrp-s13-71671014): parse_error death. Writers emitted massive full-rewrites instead of unified diffs.
- c0022 (cvrp-s13-71671014): NameError death (`_unpack` undefined). Attempted to add strict cost comparisons.
- c0010 (run cvrp-s17-78412700): NameError death. Emitted only `def _py_cross_exchange(...)` instead of a unified diff. Attempted inter-route CROSS-exchange (swap segments of length 1-2 between routes).
- c0001 (run cvrp-s19-83885116): parse_error death. Attempted budget-limited KNN-filtered inter-route 2-opt* as a final polish for long routes.
- c0005 (run cvrp-s19-83885116): parse_error death. Attempted numpy-vectorized nearest-depot-regrouping overlay to insert depot-far endpoints into KNN neighbor routes.
- c0002r2 (cvrp-s23-89572315): parse_error death (empty code string). Attempted budget-safe second C LNS call with a different seed for large instances.
- c0004 (cvrp-s23-89572315): SyntaxError death. Attempted polar-angle sweep construction alternative, picking the cheaper of {savings, sweep} to seed the C kernel.
- c0005r1 (cvrp-s23-89572315): SyntaxError death. Attempted to remove broken `try_intra_3opt` to save overlay time, but accidentally deleted the entire C kernel string.
- c0017 (cvrp-s23-89572315): TypeError death. Attempted multi-seed construction selection using `rng.uniform`.
- c0021 (cvrp-s23-89572315): NameError death (`solve` is not defined). Massive truncation left `solve` completely undefined.
- c0025r2 (cvrp-s23-89572315): parse_error death (empty code). Attempted inter-route Or-opt(4) KNN-anchored insertion.
- c0003r1 (run cvrp-s29-99479842): NameError death (`net_fwd` undefined globally). Attempted adaptive early termination of the C call with plateau detection.
- c0006r2 (run cvrp-s29-99479842): parse_error death (empty code). Attempted bug fixes in `_py_segment_shift` alongside adding an intra-route Or-opt.
- c0022r2 (run cvrp-s31-5170619): parse_error death (empty code string). Attempted to analyze and fix `_py_segment_shift` `net`/`net_fwd` bug.
- c0023 (run cvrp-s31-5170619): SyntaxError death. Emitted raw C snippets instead of a unified diff. Attempted C kernel visibility tweaks / architectural exploration.
- c0036, c0038, c0040 (run cvrp-s31-5170619): ValueError death (`too many values to unpack`). Attempted to restore the c0035 architecture (stripping bad overlays) but mistakenly unpacked 2 values from `_pack` which returns 3.

## Why it is worth keeping
- These failed for infrastructure reasons, not on merit; losing them silently biases the record. 
- However, cvrp-s11-66566581 tested the large-n recreate hypotheses and they regressed, lowering expected value.

## Verdict
promising-to-neutral, mostly untested. Priority order: (1) c0003's SA claim, verification only. (2) c0010's inter-route cross-exchange. (3) c0005's depot-distance regrouping. (4) c0003r1's C plateau detection (needs proper full-diff formatting). (5) c0024r1's ejection-chain Or-opt (truncated but viable logic).
