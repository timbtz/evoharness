# Local search — the fixed polish, and going beyond it

## What the harness already does
Every candidate tour gets the SAME polish: 2-opt, first-improvement scan order,
at most 3 full passes, epsilon 1e-9, then closed-tour length is measured. A 2-opt
move replaces edges (a,b) and (c,e) with (a,c) and (b,e), i.e. reverses the segment
between b and c. Consequences:

- Reimplementing plain 2-opt inside build_tour is wasted effort — it adds nothing
  the polish won't do, and the polish is capped at 3 passes either way.
- The polish is NOT run to convergence and is order-dependent: constructions that
  leave many crossings may not get them all removed in 3 passes.
- Your leverage: produce tours whose remaining defects 2-opt CAN fix in 3 passes
  (local crossings) and avoid defects it cannot (see below).

## What 2-opt cannot fix
- Misplaced single nodes / short segments: moving one node elsewhere is an Or-opt
  (segment insertion) move, not a 2-opt move. NN tours are full of these.
- 3-opt-only defects (segment reversal + reinsertion combinations).

## Or-opt (worth doing inside build_tour)
Try relocating segments of length 1, 2, 3: remove t[i..i+L-1], reinsert (possibly
reversed) between some (j, j+1) if it shortens the tour. O(n^2) per pass, a few
passes suffice. Or-opt + the harness's 2-opt approximates 3-opt quality: typically
1-3 percentage points better final gap than construction alone.

## Combining with restarts
Cheap loop that works well: for each seeded restart, construct, run 1-2 Or-opt
passes, keep best by true closed length (compute with d[t, np.roll(t, -1)].sum()).
Score candidates by *their own* length — the harness polish is applied after, and
better pre-polish tours almost always stay better post-polish.

## Budget notes
The Python-level polish in the harness costs O(n^2) per pass and is charged to your
timeout. At n=200: harness polish + NN baseline take roughly 1-2 s per instance;
keep your own construction under ~5 s per instance to stay clear of the 60-90 s
split budget.
