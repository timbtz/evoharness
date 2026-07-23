# C kernel rewrites, ports, and consolidations

Rewriting, porting, or structurally consolidating the C kernel (or the Python<->C architecture) failed in all attempts: crash, silent no-output, or heavy regression — never a gain.

## How it was tried
- c0001 (-6.39): "memmove-free" swap/tails, multi-direction Or-opt, K=30.
- c0005/c0007 (no output): consolidated the multi-call bridge into one C call.
- c0018/c0018r1 (compile fail / -6.12): clean do_move rewrite itself was algorithmically worse.
- c0020 (crash): freed 2D `RPRE` prefix array wrongly.
- c0021 (-2.53) / c0022 (-0.159): full granular neighborhood rebuilds that lost incumbent density.
- c0068 (-6.84): replaced O(n²) greedy insertion in recreate with KNN-filtered insertion.
- c0069/c0070 (no output): added route-minimization penalty pre-pass / dedicated `polish()` function in C.
- c0011/c0023 (cvrp-s11-66566581): added new C `lns_batch`/`two_phase_lns` entry points. Catastrophic regression or silent failure.
- c0004 (run cvrp-s19-83885116): added C-side `oropt_polish` entry point for exhaustive intra-route full-scan. Train -0.1635.
- c0008 (run cvrp-s19-83885116): added C-side `polish()` for intra-route Or-opt/2-opt replacing tier 1 Python. Train -0.1230.
- c0011 (run cvrp-s19-83885116): added C-side `polish()` between C LNS and overlay. Timed out (>35s compile/execution death).

## Why it failed
- The incumbent kernel is dense, interlocked (RID/POS/RPRE bookkeeping, granular KNN moves, SA-lite LNS); rewrites reliably drop implicit behavior they don't know is load-bearing.
- Big diffs maximize exposure to the C failure modes catalogued in implementation-insights/c-failure-modes.md.
- Adding new standalone C functions (c0070, c0011, c0023) bypasses Python ctypes architecture safely but suffers silent failures (hangs/aborts) due to untested memory management.

## Verdict
refuted. Do not rewrite or port the C kernel, replace recreate heuristics, or add standalone C search passes / entry points.
