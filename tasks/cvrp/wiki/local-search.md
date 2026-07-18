# Local search

All deltas below are exact over integer `dist` (no epsilon needed); apply a move
iff delta < 0. `D(i,j)` = dist[i,j]. `a,b` denote a node's current route-neighbors
(depot 0 if at a route end).

## Intra-route 2-opt
Reverse segment seq[i..j] within one route: replaces edges (a,seq[i]) and
(seq[j],b) with (a,seq[j]) and (seq[i],b).
delta = D(a,seq[j]) + D(seq[i],b) - D(a,seq[i]) - D(seq[j],b).
This is what the seed's `two_opt` already does (tasks/cvrp/seed.py) — intra-route only.

## Or-opt (segment relocate, length 1-3) / inter-route relocate
Remove chain [p..p+L-1] (L in {1,2,3}) between neighbors a,b:
removal_gain = D(a,seq[p]) + D(seq[p+L-1],b) - D(a,b).
Reinsert between (x,y), same or different route, forward or reversed:
insert_cost = D(x,seq[p]) + D(seq[p+L-1],y) - D(x,y)              [forward]
            = D(x,seq[p+L-1]) + D(seq[p],y) - D(x,y)              [reversed]
Apply if insert_cost < removal_gain. L=1, single target route = the seed's
`relocate`; generalize to L<=3 and this subsumes it. Capacity precheck BEFORE
scanning target positions: skip route r2 entirely unless load[r2]+demand(chain)
<= cap — O(1) reject instead of an O(route length) scan.

## Inter-route swap
Exchange c1 (route r1) and c2 (route r2):
delta = [D(a1,c2)+D(c2,b1)+D(a2,c1)+D(c1,b2)] - [D(a1,c1)+D(c1,b1)+D(a2,c2)+D(c2,b2)]
Capacity check both directions: load[r1]-demand[c1]+demand[c2] <= cap AND
load[r2]-demand[c2]+demand[c1] <= cap.

## 2-opt* (tail exchange — strongest inter-route move, missing from the seed)
Cut edge (a,b) in R1 after position i, edge (c,e) in R2 after position j;
reconnect a-e and c-b — swap the TAILS (everything after each cut) between the
two routes. No segment reversal needed (unlike 2-opt): each route keeps its
internal visit order, just spliced at a different join point.
delta = D(a,e) + D(c,b) - D(a,b) - D(c,e).
Capacity check via precomputed load-prefix sums per route (O(1) per pair):
load(R1 head)+load(R2 tail) <= cap AND load(R2 head)+load(R1 tail) <= cap.
This move reshapes route boundaries that relocate/swap/2-opt cannot reach and is
typically the single biggest quality jump addable to the seed's kernel.

## Speed tricks
- **Candidate lists**: precompute per node its K~10-20 nearest neighbors
  (argpartition per dist row, O(n^2) once); restrict move search to pairs (i,j)
  with j in i's list — turns an O(n^2) sweep into O(n·K). Nearly all improving
  CVRP moves connect geometrically close nodes; little quality lost.
- **Don't-look bits**: one bit/customer, all set initially. Only scan from
  active customers; clear a bit when no improving move is found from it;
  re-set bits of every customer whose edges just changed. Cuts rescans 5-20x
  near a local optimum.
- **First- vs best-improvement**: first-improvement (apply first delta<0,
  restart scan) converges faster wall-clock, fits more passes in `tlimit`.
  Best-improvement needs fewer iterations but each costs O(n·K) — prefer
  first-improvement under a hard budget, best-improvement only for short
  late-stage polish.
- **Termination**: `dist` is int32, every applied move has delta <= -1 (no
  float drift), total cost is bounded below, so `while(improved)` has a finite
  move count — only a time cap is needed as a safety net, not an iteration cap.
