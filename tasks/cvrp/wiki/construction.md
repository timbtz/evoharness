# Construction heuristics

## Clarke-Wright savings (the seed's approach)
s(i,j) = d(0,i) + d(0,j) - d(i,j): distance saved by serving i and j on one route
via the edge (i,j) instead of two separate depot round trips. Sort descending;
merge routes ending at i and starting at j into ...-i-j-... if (a) i and j are
currently route ENDPOINTS (adjacent to depot, not interior — an interior node
already has two route-neighbors, nothing left to merge), (b) the two routes are
distinct, (c) combined load <= capacity. O(n^2 log n) for the sort, O(n^2) merges.

## λ-parameterized savings (route-shape bias)
s(i,j) = d(0,i) + d(0,j) - λ·d(i,j), λ in [0.6, 1.4]. λ only rescales the
subtracted term: λ<1 makes merges look less attractive => more, shorter routes
(good for tight capacity / clustered demand, P set). λ>1 inflates savings =>
aggressive merging => fewer, longer routes (good when capacity is slack).
Sweeping λ over a few values gives several structurally different deterministic
starts almost for free — reuse as the diversity step in a multi-start scheme.

## Sweep algorithm
Sort customers by polar angle atan2(y-y0, x-x0) around the depot; walk the sorted
order, start a new route whenever the next customer would exceed capacity; order
each bin as a mini-TSP (nearest-neighbor + a couple 2-opt passes, bin size usually
<15). Weakness: angular order ignores density — on clustered instances (B set) a
sweep cut can slice through a cluster, giving noticeably worse routes than savings.

## Sequential vs parallel insertion
Sequential: grow one route to completion (insert cheapest feasible customer
repeatedly) before opening the next — fast, myopic, weak under tight capacity
(late routes get whatever's left). Parallel: keep all routes open simultaneously,
always insert the globally cheapest feasible (customer, route, position) —
better quality, same O(n^2) per pass with incremental best-insertion tracking.

## Regret-k insertion
For each unrouted customer c, find its best feasible insertion cost and its
k-th-best (k=2 or 3) across all routes and positions: regret(c) = cost_kth(c) -
cost_best(c). Insert the customer with MAXIMUM regret first, not minimum cost —
a customer with only one good slot must be placed now, before that slot is taken
by someone else; pure greedy-cheapest defers hard customers until only bad slots
remain. Wins most on tight-capacity instances (P set) where feasible slots are
scarce. O(n^2) per insertion with lazy incremental updates (only positions near
the last insert change), O(n^3) if recomputed from scratch each step.

## Typical quality (gap vs best-known)
Savings alone: ~8-12%. Sweep: comparable to savings on uniform-random (A),
worse on clustered (B). Regret-2/3 insertion: ~6-10%, best relative advantage on
P. None of these compete with local search (see local-search.md) — construction
only sets the anytime loop's starting point; spend little of the budget here.
