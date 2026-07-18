# Construction heuristics

All of these beat plain nearest neighbor (NN) on average; the harness polishes every
tour with the same fixed 2-opt, so what matters is how good the *polished* result is.
A construction that leaves few long "mistake" edges polishes much better than NN.

## Nearest neighbor (the seed)
Start at node 0, repeatedly go to the closest unvisited node. O(n^2). Weakness: the
last edges are terrible (it paints itself into corners), typically 20-25% above
optimum raw, 5-9% after 2-opt polish. Variants: best-of-n starts (run NN from every
start, keep shortest — O(n^3) is too slow for n=200; sample ~10 seeded starts).

## Greedy edge (usually the best simple construction)
Sort all n(n-1)/2 edges by length; add an edge unless it would give a node degree 3
or close a subtour early (union-find for cycle detection, degree array). Finish by
linking the two remaining degree-1 endpoints. Raw gap ~15-20%, polishes to ~3-5%.
O(n^2 log n) — fits the budget at n=200.

## Insertion family
Keep a subtour, repeatedly insert a node k at the position minimizing
d[i,k]+d[k,j]-d[i,j]. Cheapest insertion: pick the node with globally cheapest
insertion. Farthest insertion: insert the node farthest from the subtour —
counterintuitively strong (~7-12% raw, often best-in-family after polish).
Nearest insertion is weakest. O(n^2) with incremental min-distance arrays.

## Clarke-Wright savings
Start with star tours through a hub h (often the centroid-nearest node); merge routes
by descending savings s(i,j) = d[i,h]+d[h,j]-d[i,j], respecting degree/cycle limits
like greedy edge. Comparable to greedy edge; hub choice matters — try 2-3 hubs.

## Randomized restarts
Any of the above + seeded randomization (e.g. pick among top-3 candidate moves with
rng = np.random.default_rng(0)) x N restarts, keep the tour that is shortest AFTER a
quick local check. Restarts are the cheapest reliable improvement — budget them: at
n=200 a Python-loop construction runs ~10-50 ms, so 10-30 restarts are safe.
