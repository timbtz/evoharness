# Anytime orchestration

## Best-so-far discipline
Track `(best_routes, best_cost)` separately from the working candidate; update
only on strict improvement; return `best_*` at the end, never the current
working solution — a perturbation step easily leaves it worse than best, and
returning it forfeits banked quality. The seed does this correctly (`best,
best_c` in `solve()`); preserve the pattern.

## Time discipline
- Reserve ~0.3s margin from `deadline` (route list construction, harness JSON
  serialization); check `time.monotonic()` before any operation whose worst
  case could overrun. Harness hard-fails (-inf) only past `budget + 1.0s`.
- Python CANNOT interrupt a running ctypes call — it blocks until the C
  function returns. Pass a `tlimit` INTO every long C call and poll
  `clock_gettime(CLOCK_MONOTONIC, ...)` inside the C loop (seed's `now()`).
- Shrink `tlimit` as the deadline nears (seed: `min(remaining - 0.3, 1.0)`) so
  one call can't consume the whole remaining budget on a slow sweep.

## Perturbation strategies
- **Random ejection** (seed): remove k random customers, reinsert greedily at
  cheapest feasible slot. Cheap, weak — ejected customers are unrelated, so
  repair quality is low.
- **Segment / route ruin**: eject an entire short route, or a contiguous
  visit-order segment — bigger, structured perturbation, escapes deeper
  local optima than single-customer ejection.

## LNS ruin & recreate
Destroy 10-30% of customers via **Shaw relatedness**: r(i,j) small (related)
when d(i,j) is small (optionally + |demand[i]-demand[j]|); pick a random seed
customer, repeatedly eject its most-related remaining customers so the removed
set is spatially/demand clustered — repairs better than uniform-random removal.
Recreate with regret-k insertion (construction.md): cheap on a small set, much
stronger than greedy reinsert. Accept equal-or-better always; for exploration,
accept worse with probability exp(-delta/T) (SA-lite, T decaying) or a
record-to-record threshold (accept if within 1-2% of best).

## Restart schedules
Spend early budget on diversity: build several starts (lambda-savings sweep
across lambda in {0.6,0.8,1.0,1.2,1.4}, regret insertion, angular sweep — see
construction.md), converge each with one quick local-search pass, keep the
best as anchor. Spend the rest on intensification (LNS/perturbation from that
anchor). n<=65 (train/val, 3s): convergence takes well under a second, so most
of the budget IS the perturbation loop — iteration count/ruin quality dominate.
n=101-303 (5-6s): initial convergence itself takes a real slice — kernel speed
starts to matter too (local-search.md).

## Division of labor: Python vs C
Python decides WHAT to try next (perturbation choice, which routes, when to
switch diversify->intensify) — cheap control logic. C executes the
O(n^2)-per-sweep work (2-opt/relocate/swap/2-opt* convergence) in short
`tlimit` slices rather than one long call, so Python regains control between
slices to check the deadline and pick the next move.

## Compile once
`compile_c` costs ~0.3-0.8s (gcc -O3 -march=native). Instances in a split run
in the SAME process (task.py's per-split loop calls `solve()` repeatedly) —
put every kernel in one C source, compile on the first `solve()` call, cache
the `CDLL` in a module global. Only the first train instance pays the compile
cost; recompiling per instance burns a meaningful share of the 3s budget.
