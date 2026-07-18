"""Seed artifact: parallel Clarke-Wright savings construction (Clarke & Wright
1964) in Python, then a C engine — granular local search over K-nearest-neighbor
candidate lists with don't-look bits (Toth & Vigo 2003) running relocate, Or-opt
(Or 1976), swap, 2-opt and 2-opt*, inside an LNS ruin-and-recreate loop (Shaw
1998 related removal, regret-2 reinsertion, SA-lite acceptance). The C engine is
compiled once per process via compile_c and driven in time slices so the
per-instance deadline is always respected; Python keeps the best verified
solution after every slice."""
import ctypes
import random
import time

import numpy as np

C_KERNEL = __KERNEL_C__

_LIB = None  # compile once, reuse across the instances of one evaluation run


def _lib(compile_c):
    global _LIB
    if _LIB is None:
        _LIB = compile_c(C_KERNEL)
        I = ctypes.POINTER(ctypes.c_int)
        _LIB.lns.argtypes = [I, I, ctypes.c_int, ctypes.c_int,
                             I, I, I, I, I, I,
                             ctypes.c_double, ctypes.c_double, ctypes.c_double,
                             ctypes.c_ulonglong]
        _LIB.lns.restype = None
    return _LIB


def _savings_routes(dist, demand, cap):
    """Parallel Clarke-Wright savings (1964): merge routes at endpoints by
    descending savings s(i,j) = d(0,i) + d(0,j) - d(i,j)."""
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


def _pack(routes, n):
    """Routes -> flat (seq, rend, k) arrays as the C engine expects."""
    seq, rend = np.zeros(n, np.int32), np.zeros(n, np.int32)
    off = 0
    for k, r in enumerate(routes):
        seq[off:off + len(r)] = r
        off += len(r)
        rend[k] = off
    return seq, rend, len(routes)


def _unpack(seq, rend, k):
    out, s = [], 0
    for r in range(k):
        e = int(rend[r])
        out.append([int(x) for x in seq[s:e]])
        s = e
    return out


def _valid(routes, demand, cap, n):
    """Cheap host-side sanity net: partition of 1..n-1, loads within capacity."""
    flat = [c for r in routes for c in r]
    if sorted(flat) != list(range(1, n)):
        return False
    return all(sum(int(demand[c]) for c in r) <= cap for r in routes)


def solve(coords, dist, demand, capacity, deadline, compile_c):
    rng = random.Random(0)
    lib = _lib(compile_c)
    dist = np.ascontiguousarray(dist, dtype=np.int32)
    dem = np.ascontiguousarray(demand, dtype=np.int32)
    n, cap = len(dem), int(capacity)
    if n <= 2:
        return [[1]] if n == 2 else []

    routes = _savings_routes(dist, dem, cap)
    best = [list(r) for r in routes]
    cseq, crend, ck = _pack(routes, n)       # LNS current solution
    bseq, brend, bk = _pack(routes, n)       # best-so-far solution
    ckv, bkv = ctypes.c_int(ck), ctypes.c_int(bk)
    I = ctypes.POINTER(ctypes.c_int)

    # Time-sliced anytime loop: each slice hands the C engine a bounded budget
    # and global-time fractions so its annealing schedule cools continuously;
    # after each slice the improved best is verified and kept in Python.
    t0 = time.monotonic()
    total = max(deadline - 0.35 - t0, 0.01)
    while True:
        left = deadline - 0.35 - time.monotonic()
        if left <= 0.05:
            break
        sl = min(left, 1.5)
        e0 = time.monotonic() - t0
        lib.lns(dist.ctypes.data_as(I), dem.ctypes.data_as(I), n, cap,
                cseq.ctypes.data_as(I), crend.ctypes.data_as(I), ctypes.byref(ckv),
                bseq.ctypes.data_as(I), brend.ctypes.data_as(I), ctypes.byref(bkv),
                ctypes.c_double(sl), ctypes.c_double(min(e0 / total, 1.0)),
                ctypes.c_double(min((e0 + sl) / total, 1.0)),
                ctypes.c_ulonglong(rng.getrandbits(63) | 1))
        cand = _unpack(bseq, brend, bkv.value)
        if _valid(cand, dem, cap, n):
            best = cand
    return best
