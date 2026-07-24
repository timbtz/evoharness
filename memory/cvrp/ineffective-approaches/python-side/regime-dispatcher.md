# Regime dispatcher: choose polish strategy by estimated customers-per-route

Dispatching the Python overlay strategy dynamically based on the estimated customers-per-route (n/k) was attempted across multiple runs. It is now refuted: the n/k gating itself provides no measurable architectural advantage on train/val sets, and stealing time from the overlay to give to C triggers time-starvation.

## How it was tried
- c0001 (run cvrp-s17-78412700): attempted to dispatch short-route instances (n/k < 8) to a cheaper Or-opt(1)+2-opt overlay, skipping Or-opt(2,3) and segment-shift. Long routes (>10) got full segment-shift. Train -0.1279, val -0.2302 (tie).
- c0004/c0005 (run cvrp-s17-78412700): dispatched unified overlay to all instances, segment-shift gated for `n/k > 10`. Train -0.1364 / -0.1317. Rejected.
- c0006 (run cvrp-s17-78412700): regime-aware budget split (reserve 0.12s for long-route overlay, give C full budget for short routes). Train -0.1227, val -0.2766. Rejected.
- c0001 (run cvrp-s31-5170619): short-route regime includes segment-shift(max_seg=2) to catch stranded customers. Train -0.0774, val -0.2942.
- c0018 (run cvrp-s31-5170619): SKIP the Python overlay entirely for short routes (cust_per_route ≤ 6) to give C the full budget. Train -0.098, val -0.233.

## Why it failed
- **Code Inspection Findings (c0001):** The regime dispatcher logic was heavily compromised by bugs in the initial attempts. The long-route overlay branch was completely truncated, defaulting to no overlay. 
- **Outcome vs. Prediction:** Writers predicted that short routes would benefit from cheaper overlays saving budget. The mechanism was refuted: stealing time from the overlay to give to the C LNS just triggers time-starvation (c0006) or converges identically (c0004/c0005), as the Python overlay is already strictly-improving and cheap (~0.01s). 
- c0018 proved that even completely removing the overlay for short routes makes no difference, confirming the overlay is cheap and skipping it unlocks no meaningful C LNS gradient.

## Verdict
exhausted. Do not use regime dispatchers to split overlay architecture or dynamically steal budget for C LNS. The unified `Or-opt(1-3)+intra-2-opt+swap+segment-shift` overlay is safe, cheap across all n/k regimes, and cannot be optimized further via regime gating.
