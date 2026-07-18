"""Seed artifact: Clarke-Wright savings construction (Python) + a C local-search
kernel (intra-route 2-opt, relocate, inter-route swap) compiled via compile_c,
driven by an anytime perturb-and-reoptimize loop until the deadline."""
import ctypes
import random
import time

import numpy as np

C_KERNEL = r"""
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

#define D(i, j) dist[(long)(i) * n + (j)]

/* Routes in one flat array: seq = all customers in visit order, rend[r] =
   exclusive end of route r; depot 0 implicitly starts/ends every route.
   First-improvement sweeps until converged or tlimit seconds pass. Integer
   distances => every applied move gains >= 1, so convergence is finite. */

static int two_opt(const int *dist, int n, int *seq, const int *rend, int k,
                   double tend) {
    int any = 0;
    for (int r = 0, s = 0; r < k; s = rend[r++]) {
        int e = rend[r];
        if (now() > tend) return any;
        for (int i = s; i < e - 1; i++)
            for (int j = i + 1; j < e; j++) {
                int a = i == s ? 0 : seq[i - 1], b = j == e - 1 ? 0 : seq[j + 1];
                if (D(a, seq[j]) + D(seq[i], b) < D(a, seq[i]) + D(seq[j], b)) {
                    for (int x = i, y = j; x < y; x++, y--) {
                        int t = seq[x]; seq[x] = seq[y]; seq[y] = t;
                    }
                    any = 1;
                }
            }
    }
    return any;
}

static int relocate(const int *dist, const int *demand, int n, int cap,
                    int *seq, int *rend, int *load, int k, double tend) {
    int m = rend[k - 1];
    for (int r = 0, s = 0; r < k; s = rend[r++]) {
        int e = rend[r];
        if (now() > tend) return 0;
        for (int p = s; p < e; p++) {
            int c = seq[p];
            int a = p == s ? 0 : seq[p - 1], b = p == e - 1 ? 0 : seq[p + 1];
            int gain = D(a, c) + D(c, b) - D(a, b);
            if (gain <= 0) continue;
            for (int r2 = 0, s2 = 0; r2 < k; s2 = rend[r2++]) {
                int e2 = rend[r2];
                if (r2 != r && load[r2] + demand[c] > cap) continue;
                for (int q = s2; q <= e2; q++) {
                    if (r2 == r && (q == p || q == p + 1)) continue;
                    int x = q == s2 ? 0 : seq[q - 1], y = q == e2 ? 0 : seq[q];
                    if (D(x, c) + D(c, y) - D(x, y) < gain) {
                        memmove(seq + p, seq + p + 1, (size_t)(m - p - 1) * sizeof(int));
                        for (int t = r; t < k; t++) rend[t]--;
                        int qq = q > p ? q - 1 : q;
                        memmove(seq + qq + 1, seq + qq, (size_t)(m - 1 - qq) * sizeof(int));
                        seq[qq] = c;
                        for (int t = r2; t < k; t++) rend[t]++;
                        load[r] -= demand[c]; load[r2] += demand[c];
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

static int swap_op(const int *dist, const int *demand, int n, int cap,
                   int *seq, const int *rend, int *load, int k, double tend) {
    int any = 0;
    for (int r = 0, s = 0; r < k; s = rend[r++]) {
        if (now() > tend) return any;
        for (int r2 = r + 1, s2 = rend[r]; r2 < k; s2 = rend[r2++])
            for (int p = s; p < rend[r]; p++)
                for (int q = s2; q < rend[r2]; q++) {
                    int c1 = seq[p], c2 = seq[q];
                    if (load[r] - demand[c1] + demand[c2] > cap ||
                        load[r2] - demand[c2] + demand[c1] > cap) continue;
                    int a1 = p == s ? 0 : seq[p - 1];
                    int b1 = p == rend[r] - 1 ? 0 : seq[p + 1];
                    int a2 = q == s2 ? 0 : seq[q - 1];
                    int b2 = q == rend[r2] - 1 ? 0 : seq[q + 1];
                    if (D(a1, c2) + D(c2, b1) + D(a2, c1) + D(c1, b2) <
                        D(a1, c1) + D(c1, b1) + D(a2, c2) + D(c2, b2)) {
                        seq[p] = c2; seq[q] = c1;
                        load[r] += demand[c2] - demand[c1];
                        load[r2] += demand[c1] - demand[c2];
                        any = 1;
                    }
                }
    }
    return any;
}

void improve(const int *dist, const int *demand, int n, int cap,
             int *seq, int *rend, int k, double tlimit) {
    if (k <= 0) return;
    double tend = now() + tlimit;
    int load[k];
    for (int r = 0, s = 0; r < k; s = rend[r++]) {
        load[r] = 0;
        for (int p = s; p < rend[r]; p++) load[r] += demand[seq[p]];
    }
    int improved = 1;
    while (improved && now() < tend) {
        improved = two_opt(dist, n, seq, rend, k, tend);
        improved |= relocate(dist, demand, n, cap, seq, rend, load, k, tend);
        improved |= swap_op(dist, demand, n, cap, seq, rend, load, k, tend);
    }
}
"""


def _savings_routes(dist, demand, cap):
    """Parallel Clarke-Wright: merge routes at endpoints by descending savings."""
    n = len(demand)
    routes = {c: [c] for c in range(1, n)}   # route id -> customer list
    rid = {c: c for c in range(1, n)}        # customer -> route id
    load = {c: int(demand[c]) for c in range(1, n)}
    cust = np.arange(1, n)
    iu, ju = np.triu_indices(n - 1, k=1)
    a, b = cust[iu], cust[ju]
    sav = dist[0][a] + dist[0][b] - dist[a, b]
    order = np.argsort(-sav, kind="stable")
    for i, j, s in zip(a[order], b[order], sav[order]):
        if s <= 0:
            break
        ri, rj = rid[int(i)], rid[int(j)]
        if ri == rj or load[ri] + load[rj] > cap:
            continue
        A, B = routes[ri], routes[rj]
        if A[-1] != i and A[0] == i:
            A.reverse()
        if B[0] != j and B[-1] == j:
            B.reverse()
        if A[-1] != i or B[0] != j:
            continue                          # i or j is interior — cannot merge
        A.extend(B)
        for c in B:
            rid[c] = ri
        load[ri] += load[rj]
        del routes[rj], load[rj]
    return list(routes.values())


def _cost(routes, dist):
    total = 0
    for r in routes:
        prev = 0
        for c in r:
            total += int(dist[prev, c]); prev = c
        total += int(dist[prev, 0])
    return total


def _perturb(routes, demand, cap, rng):
    """Ruin & recreate, tiny: eject 3 random customers, reinsert feasibly at random."""
    routes = [list(r) for r in routes]
    flat = [c for r in routes for c in r]
    picks = rng.sample(flat, min(3, len(flat)))
    for c in picks:
        for r in routes:
            if c in r:
                r.remove(c); break
    routes = [r for r in routes if r]
    loads = [sum(int(demand[c]) for c in r) for r in routes]
    for c in picks:
        feas = [t for t in range(len(routes)) if loads[t] + int(demand[c]) <= cap]
        if feas:
            t = rng.choice(feas)
            routes[t].insert(rng.randrange(len(routes[t]) + 1), c)
            loads[t] += int(demand[c])
        else:
            routes.append([c]); loads.append(int(demand[c]))
    return routes


def solve(coords, dist, demand, capacity, deadline, compile_c):
    rng = random.Random(0)
    lib = compile_c(C_KERNEL)
    I = ctypes.POINTER(ctypes.c_int)
    lib.improve.argtypes = [I, I, ctypes.c_int, ctypes.c_int, I, I,
                            ctypes.c_int, ctypes.c_double]
    lib.improve.restype = None
    dist = np.ascontiguousarray(dist, dtype=np.int32)
    dem = np.ascontiguousarray(demand, dtype=np.int32)
    n, cap = len(dem), int(capacity)

    def optimize(routes, tlimit):
        routes = [r for r in routes if r]
        seq = np.array([c for r in routes for c in r], dtype=np.int32)
        rend = np.cumsum([len(r) for r in routes]).astype(np.int32)
        lib.improve(dist.ctypes.data_as(I), dem.ctypes.data_as(I), n, cap,
                    seq.ctypes.data_as(I), rend.ctypes.data_as(I), len(routes),
                    ctypes.c_double(max(tlimit, 0.0)))
        out, s = [], 0
        for e in rend:
            if e > s:
                out.append([int(x) for x in seq[s:e]])
            s = int(e)
        return out

    best = optimize(_savings_routes(dist, dem, cap),
                    deadline - time.monotonic() - 0.3)
    best_c = _cost(best, dist)
    while time.monotonic() < deadline - 0.35:
        cand = _perturb(best, dem, cap, rng)
        cand = optimize(cand, min(deadline - time.monotonic() - 0.3, 1.0))
        c = _cost(cand, dist)
        if c < best_c:
            best, best_c = cand, c
    return best
