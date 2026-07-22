# C kernel rewrites, ports, and consolidations

Rewriting, porting, or structurally consolidating the C kernel (or the Python<->C architecture) failed in all 8 attempts: crash, silent no-output, or heavy regression — never a gain.

## How it was tried
- c0001 (-6.39): "memmove-free" swap/tails, multi-direction Or-opt, K=30, Python savings parallelism — introduced the savings i/j-mutation bug (fixed by c0002, see successful-patterns/surgical-bug-fixes.md).
- c0005 (no output): moved Clarke-Wright construction into C + Or-opt-with-reversal — never produced a solution.
- c0007 (no output): consolidated the multi-call bridge into one `solve_cvrp` C call with neighbor-guided recreate.
- c0006 (SyntaxError): same single-kernel idea, but the writer emitted mangled code (see implementation-insights/generation-parse-failures.md).
- c0018 (compile fail: helpers `pos_out`/`seg_in` deleted but still called) -> repair c0018r1 compiled but scored -6.12: the "clean do_move" rewrite itself was algorithmically worse.
- c0020 (crash `munmap_chunk(): invalid pointer`): full pipeline port to C; freed the 2D `RPRE` prefix array wrongly.
- c0021 (-2.53): rewrite avoiding the 2D `RPRE` array + "true Or-opt" — ran, but lost the incumbent's algorithmic density.
- c0022 (-0.159): rebuilt local search from scratch as a full granular neighborhood with don't-look bits — best rewrite result, still a regression.

## Why it failed
- The incumbent kernel is dense, interlocked (RID/POS/RPRE bookkeeping, granular KNN moves, SA-lite LNS); rewrites reliably drop implicit behavior they don't know is load-bearing.
- Big diffs maximize exposure to the C failure modes catalogued in implementation-insights/c-failure-modes.md.
- Even when a rewrite compiles and runs (c0021, c0022, c0018r1), it re-derives years of tuning from scratch inside a 5s budget and loses.

## Verdict
refuted. Do not rewrite or port the C kernel, and do not restructure the Python<->C call architecture as a rewrite. The one architecture idea never actually tested (single long C call preserving SA state, c0054/c0037/c0009 — all parse-failed) is tracked in new-ideas/persistent-c-search-state.md.
