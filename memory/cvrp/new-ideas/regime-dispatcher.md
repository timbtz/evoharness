# Regime dispatcher: choose polish strategy by estimated customers-per-route

Evaluated for the first time in cvrp-s17-78412700 (c0001). By inspecting the estimated customers-per-route (n/k) at runtime, it tries to dispatch the Python overlay strategy dynamically. The initial attempt was accepted but only as a tie/noise result, and its implementation was marred by a bug.

## How it was tried
- c0001 (run cvrp-s17-78412700): attempted to dispatch short-route instances (n/k < 8) to a cheaper Or-opt(1)+2-opt overlay, skipping Or-opt(2,3) and segment-shift. Long routes (>10) got full segment-shift. Accepted with train -0.1279 / val -0.2302 (treat as noise-band tie).
- c0004 (run cvrp-s17-78412700): dispatched c0035's unified overlay to all instances but added `_py_segment_shift` only if `cust_per_route > 10`. Train -0.1364, val -0.3013. Rejected.
- c0005 (run cvrp-s17-78412700): applied unified overlay for all instances, segment-shift gated for `n/k > 10`. Train -0.1317, val -0.2700. Rejected.
- c0006 (run cvrp-s17-78412700): regime-aware budget split (reserve 0.12s for long-route overlay, give C full budget for short routes). Train -0.1227, val -0.2766. Rejected.

## Why it failed / has not worked yet
- **Code Inspection Findings (c0001):** The single-call C LNS restoration worked, but the regime dispatcher logic was heavily compromised by bugs. The long-route overlay branch was completely truncated (lost in generation), defaulting to no overlay. The short-route branch was also truncated.
- **Outcome vs. Prediction:** Writers predicted that short routes would benefit from cheaper overlays saving budget. The mechanism was refuted: stealing time from the overlay to give to the C LNS just triggers time-starvation (c0006) or converges identically (c0004/c0005), as the Python overlay is already strictly-improving and cheap (~0.01s). The n/k gating itself provides no measurable architectural advantage on train/val sets.

## Verdict
exhausted. Do not use regime dispatchers to split overlay architecture or dynamically steal budget for C LNS. The unified `Or-opt(1-3)+intra-2-opt+swap` overlay is safe and cheap across all n/k regimes on the train/val sets.
