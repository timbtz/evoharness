"""Anytime CVRP: parallel Clarke-Wright + granular C local search & LNS.
Pure-Python Or-opt polish added on top of the C LNS best (no C changes)."""
import ctypes, random, time
import numpy as np

C_KERNEL = r"""
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
static unsigned long long RS;
static unsigned int rnd(void) {
    RS ^= RS >> 12; RS ^= RS << 25; RS ^= RS >> 27;
    return (unsigned int)((RS * 2685821657736338717ULL) >> 32);
}
static double rndd(void) { return rnd() * (1.0 / 4294967296.0); }

static const int *DIST, *DEM;
static int N, CAP, K, R;
static int *RT, *RLEN, *RLOAD, *RID, *POS, *DLB, *ORD, *TMP, *NBR;
static int **RP, **RPRE;
#define D(a, b) DIST[(long)(a) * N + (b)]
#define AT(r, i) RT[(long)(r) * N + (i)]
#define G(r, i) RP[r][i]

static int prvn(int r, int i) { return i > 0 ? G(r, i - 1) : 0; }
static int nxtn(int r, int i) { return i < RLEN[r] - 1 ? G(r, i + 1) : 0; }
static void wake(int c) { if (c > 0) DLB[c] = 0; }

static void reload(int r) {
    int ld = 0, *pre = RPRE[r];
    for (int i = 0; i < RLEN[r]; i++) {
        int c = G(r, i);
        POS[c] = i; RID[c] = r;
        ld += DEM[c];
        pre[i] = ld;
    }
    RLOAD[r] = ld;
    RP[r] = &AT(r, 0);
}

static void pos_out(int r, int p, int L, int *buf) {
    wake(prvn(r, p));
    wake(nxtn(r, p + L - 1));
    for (int i = 0; i < L; i++) {
        RID[buf[i] = G(r, p + i)] = -1;
        RLOAD[r] -= DEM[buf[i]];
    }
    memmove(&AT(r, p), &AT(r, p + L), (size_t)(RLEN[r] - p - L) * sizeof(int));
    RLEN[r] -= L;
    reload(r);
}

static void seg_in(int r, int q, int L, const int *buf) {
    wake(q > 0 ? G(r, q - 1) : 0);
    wake(q < RLEN[r] ? G(r, q) : 0);
    memmove(&AT(r, q + L), &AT(r, q), (size_t)(RLEN[r] - q) * sizeof(int));
    for (int i = 0; i < L; i++) AT(r, q + i) = buf[i];
    RLEN[r] += L;
    reload(r);
    for (int i = 0; i < L; i++) wake(buf[i]);
}

static void rev_seg(int r, int i, int j) {
    wake(prvn(r, i)); wake(G(r, i)); wake(G(r, j)); wake(nxtn(r, j));
    while (i < j) {
        int t = AT(r, i);
        AT(r, i) = AT(r, j);
        AT(r, j) = t;
        i++; j--;
    }
    reload(r);
}

static void tails(int r, int p, int r2, int q) {
    int L1 = RLEN[r] - p - 1, L2 = RLEN[r2] - q - 1;
    memcpy(TMP, &AT(r, p + 1), (size_t)L1 * sizeof(int));
    memmove(&AT(r, p + 1), &AT(r2, q + 1), (size_t)L2 * sizeof(int));
    memcpy(&AT(r2, q + 1), TMP, (size_t)L1 * sizeof(int));
    RLEN[r] = p + 1 + L2;
    RLEN[r2] = q + 1 + L1;
    reload(r); reload(r2);
}

static void build_knn(void) {
    for (int c = 1; c < N; c++) {
        for (int u = 0; u < N; u++) TMP[u] = 0;
        for (int t = 0; t < K; t++) {
            int bu = -1, bd = 0;
            for (int u = 1; u < N; u++) {
                if (u == c || TMP[u]) continue;
                int dd = D(c, u);
                if (bu < 0 || dd < bd) { bu = u; bd = dd; }
            }
            NBR[c * K + t] = bu;
            TMP[bu] = 1;
        }
    }
}

static long long sol_cost(void) {
    long long s = 0;
    for (int r = 0; r < R; r++) {
        int pr = 0;
        for (int i = 0; i < RLEN[r]; i++) {
            int c = G(r, i);
            s += D(pr, c);
            pr = c;
        }
        s += D(pr, 0);
    }
    return s;
}

static int flatten(int *seq, int *rend) {
    int k = 0, off = 0;
    for (int r = 0; r < R; r++) {
        if (!RLEN[r]) continue;
        memcpy(seq + off, &AT(r, 0), (size_t)RLEN[r] * sizeof(int));
        off += RLEN[r];
        rend[k++] = off;
    }
    return k;
}

static void unflatten(const int *seq, const int *rend, int k) {
    R = 0;
    for (int r = 0, s = 0; r < k; r++) {
        int e = rend[r];
        if (e > s) {
            RLEN[R] = e - s;
            memcpy(&AT(R, 0), seq + s, (size_t)(e - s) * sizeof(int));
            reload(R);
            R++;
        }
        s = e;
    }
    if (R >= N) R = N - 1;
    RLEN[R] = 0;
    RLOAD[R] = 0;
    reload(R);
    R++;
}

static long long flat_cost(const int *seq, const int *rend, int k) {
    long long s = 0;
    for (int r = 0, st = 0; r < k; r++) {
        int pr = 0;
        for (int i = st; i < rend[r]; i++) {
            int c = seq[i];
            s += D(pr, c);
            pr = c;
        }
        s += D(pr, 0);
        st = rend[r];
    }
    return s;
}

static void ensure_empty(void) {
    int w = 0;
    for (int r = 0; r < R; r++) {
        if (RLEN[r]) {
            if (w != r) {
                memcpy(&AT(w, 0), &AT(r, 0), (size_t)RLEN[r] * sizeof(int));
                RLEN[w] = RLEN[r];
                reload(w);
            }
            w++;
        }
    }
    if (w >= N) w = N - 1;
    RLEN[w] = 0;
    RLOAD[w] = 0;
    reload(w);
    R = w + 1;
}

static int try_moves(int c) {
    int b1[3];
    int r = RID[c], p = POS[c];
    int a = prvn(r, p), x = nxtn(r, p);
    int x2 = x ? nxtn(r, p + 1) : 0, x3 = x2 ? nxtn(r, p + 2) : 0;
    int rem1 = D(a, c) + D(c, x) - D(a, x);
    int rem2 = x ? D(a, c) + D(x, x2) - D(a, x2) : 0;
    int rem3 = x2 ? D(a, c) + D(x2, x3) - D(a, x3) : 0;
    int q1 = DEM[c];
    int q2 = q1 + (x ? DEM[x] : 0);
    int q3 = q2 + (x2 ? DEM[x2] : 0);

    for (int t = 0; t < K; t++) {
        int u = NBR[c * K + t];
        int r2 = RID[u];
        if (r2 < 0) continue;
        int q = POS[u];
        int w = prvn(r2, q), v = nxtn(r2, q);
        int okc = (r2 == r) || (RLOAD[r2] + q1 <= CAP);

        if (okc && u != a) {
            int g = rem1 - (D(u, c) + D(c, v) - D(u, v));
            if (g > 0) {
                pos_out(r, p, 1, b1);
                seg_in(r2, POS[u] + 1, 1, b1);
                return 1;
            }
        }
        if (okc && w != c) {
            int g = rem1 - (D(w, c) + D(c, u) - D(w, u));
            if (g > 0) {
                pos_out(r, p, 1, b1);
                seg_in(r2, POS[u], 1, b1);
                return 1;
            }
        }
        if (x && u != x && u != a && (r2 == r || RLOAD[r2] + q2 <= CAP)) {
            int gf = rem2 - (D(u, c) + D(x, v) - D(u, v));
            int gr = rem2 - (D(u, x) + D(c, v) - D(u, v));
            if (gf > 0 || gr > 0) {
                pos_out(r, p, 2, b1);
                if (gr > gf) { int tt = b1[0]; b1[0] = b1[1]; b1[1] = tt; }
                seg_in(r2, POS[u] + 1, 2, b1);
                return 1;
            }
        }
        if (x2 && u != x && u != x2 && u != a && (r2 == r || RLOAD[r2] + q3 <= CAP)) {
            int gf = rem3 - (D(u, c) + D(x2, v) - D(u, v));
            int gr = rem3 - (D(u, x2) + D(c, v) - D(u, v));
            if (gf > 0 || gr > 0) {
                pos_out(r, p, 3, b1);
                if (gr > gf) { int tt = b1[0]; b1[0] = b1[2]; b1[2] = tt; }
                seg_in(r2, POS[u] + 1, 3, b1);
                return 1;
            }
        }
        if ((r2 != r && RLOAD[r] - q1 + DEM[u] <= CAP && RLOAD[r2] - DEM[u] + q1 <= CAP)
            || (r2 == r && q != p - 1 && q != p + 1)) {
            int g = D(a, c) + D(c, x) + D(w, u) + D(u, v)
                  - (D(a, u) + D(u, x) + D(w, c) + D(c, v));
            if (g > 0) {
                AT(r, p) = u;
                AT(r2, q) = c;
                reload(r);
                if (r2 != r) reload(r2);
                wake(a); wake(x); wake(w); wake(v); wake(c); wake(u);
                return 1;
            }
        }
        if (r2 == r) {
            int i = p < q ? p : q;
            int j = p < q ? q : p;
            int A = G(r, i), B = G(r, i + 1), E = nxtn(r, j);
            int g = D(A, B) + D(G(r, j), E) - D(A, G(r, j)) - D(B, E);
            if (g > 0) {
                rev_seg(r, i + 1, j);
                return 1;
            }
        } else {
            int g = D(c, x) + D(u, v) - D(c, v) - D(u, x);
            if (g > 0) {
                int plc = (p >= 0) ? RPRE[r][p] : 0;
                int plu = (q >= 0) ? RPRE[r2][q] : 0;
                if (plc + RLOAD[r2] - plu <= CAP && plu + RLOAD[r] - plc <= CAP) {
                    tails(r, p, r2, q);
                    return 1;
                }
            }
            g = D(c, x) + D(w, u) - D(c, u) - D(w, x);
            if (g > 0) {
                int plc = (p >= 0) ? RPRE[r][p] : 0;
                int plw = (q >= 1) ? RPRE[r2][q - 1] : 0;
                if (plc + RLOAD[r2] - plw <= CAP && plw + RLOAD[r] - plc <= CAP) {
                    tails(r, p, r2, q - 1);
                    return 1;
                }
            }
        }
    }

    if (rem1 > 2 * D(0, c)) {
        for (int r2 = 0; r2 < R; r2++) {
            if (!RLEN[r2]) {
                b1[0] = c;
                pos_out(r, p, 1, b1);
                seg_in(r2, 0, 1, b1);
                return 1;
            }
        }
    }
    return 0;
}

static void local_search(double tend) {
    for (int i = N - 2; i > 0; i--) {
        int j = rnd() % (i + 1);
        int t = ORD[i];
        ORD[i] = ORD[j];
        ORD[j] = t;
    }
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int t = 0; t < N - 1; t++) {
            if ((t & 15) == 0 && now() > tend) return;
            int c = ORD[t];
            if (DLB[c]) continue;
            if (try_moves(c)) improved = 1;
            else DLB[c] = 1;
        }
    }
}

static int ruin(int *rem, char *inrem) {
    int nc = N - 1, cnt = 0;
    int m = (int)((0.08 + 0.10 * rndd()) * nc);
    if (m < 5) m = 5;
    if (m > 60) m = 60;
    if (m > nc - 2) m = nc - 2;
    if (m < 1) m = 1;
    memset(inrem, 0, N);
    if (rndd() < 0.5) {
        rem[cnt] = 1 + rnd() % nc;
        inrem[rem[cnt]] = 1;
        cnt++;
        while (cnt < m) {
            int c = rem[rnd() % cnt], nc2 = 0, pick;
            for (int t = 0; t < K; t++) {
                int u = NBR[c * K + t];
                if (u > 0 && !inrem[u]) TMP[nc2++] = u;
            }
            if (nc2) pick = TMP[(int)(pow(rndd(), 3.0) * nc2)];
            else do pick = 1 + rnd() % nc; while (inrem[pick]);
            inrem[pick] = 1;
            rem[cnt++] = pick;
        }
        for (int i = 0; i < cnt; i++) {
            pos_out(RID[rem[i]], POS[rem[i]], 1, TMP);
        }
    } else {
        while (cnt < m) {
            int r = rnd() % R;
            if (!RLEN[r]) continue;
            int maxl = RLEN[r] < m - cnt ? RLEN[r] : m - cnt;
            int L = 1 + rnd() % maxl;
            int st = rnd() % (RLEN[r] - L + 1);
            for (int i = 0; i < L; i++) {
                int u = G(r, st + i);
                rem[cnt++] = u;
                inrem[u] = 1;
            }
            pos_out(r, st, L, TMP);
        }
    }
    return cnt;
}

static void recreate(int *rem, int m, char *inrem) {
    int has_empty = 0;
    for (int r = 0; r < R; r++) { if (!RLEN[r]) { has_empty = 1; break; } }
    if (!has_empty && R < N) {
        RLEN[R] = 0; RLOAD[R] = 0;
        reload(R);
        R++;
    }
    for (int left = m; left > 0; left--) {
        long long bg = -1, bb = 1LL << 40;
        int bc = -1, br = 0, bq = 0;
        for (int i = 0; i < m; i++) {
            int c = rem[i];
            if (!inrem[c]) continue;
            long long b1 = 1LL << 40, b2 = 1LL << 40;
            int r1 = 0, o1 = 0;
            for (int r = 0; r < R; r++) {
                if (RLOAD[r] + DEM[c] > CAP) continue;
                long long rb = 1LL << 40;
                int rq = 0, pr = 0;
                for (int qq = 0; qq <= RLEN[r]; qq++) {
                    int nx = qq < RLEN[r] ? G(r, qq) : 0;
                    long long inc = D(pr, c) + D(c, nx) - D(pr, nx);
                    if (inc < rb) { rb = inc; rq = qq; }
                    pr = nx;
                }
                if (rb < b1) { b2 = b1; b1 = rb; r1 = r; o1 = rq; }
                else if (rb < b2) b2 = rb;
            }
            long long reg = b2 - b1;
            if (reg > bg || (reg == bg && b1 < bb)) {
                bg = reg; bb = b1; bc = c; br = r1; bq = o1;
            }
        }
        if (bc < 0) {
            for (int r = 0; r < R; r++) {
                if (!RLEN[r]) { br = r; bq = 0; break; }
            }
            for (int i = 0; i < m; i++) {
                if (inrem[rem[i]]) { bc = rem[i]; break; }
            }
        }
        seg_in(br, bq, 1, &bc);
        inrem[bc] = 0;
        has_empty = 0;
        for (int r = 0; r < R; r++) if (!RLEN[r]) { has_empty = 1; break; }
        if (!has_empty && R < N) {
            RLEN[R] = 0;
            RLOAD[R] = 0;
            reload(R);
            R++;
        }
    }
}

void lns(const int *dist, const int *demand, int n, int cap,
         int *cseq, int *crend, int *ck,
         int *bseq, int *brend, int *bk,
         double tlimit, double f0, double f1, unsigned long long seed) {
    double t0 = now(), tend = t0 + tlimit;
    DIST = dist; DEM = demand; N = n; CAP = cap;
    K = n - 2 < 20 ? n - 2 : 20;
    if (K < 1) K = 1;
    RT = malloc((size_t)n * n * sizeof(int));
    RLEN = malloc(n * sizeof(int));
    RLOAD = malloc(n * sizeof(int));
    RID = malloc(n * sizeof(int));
    POS = malloc(n * sizeof(int));
    DLB = malloc(n * sizeof(int));
    ORD = malloc(n * sizeof(int));
    TMP = malloc(n * sizeof(int));
    NBR = malloc((size_t)n * K * sizeof(int));
    RP = malloc(n * sizeof(int *));
    RPRE = malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        RP[i] = &AT(i, 0);
        RPRE[i] = malloc(n * sizeof(int));
    }
    int *rem = malloc(n * sizeof(int));
    int *sseq = malloc(n * sizeof(int)), *srend = malloc(n * sizeof(int));
    char *inrem = malloc(n);
    RS = seed | 1;
    build_knn();
    for (int i = 0; i < N - 1; i++) ORD[i] = i + 1;
    memset(DLB, 0, n * sizeof(int));
    unflatten(cseq, crend, *ck);
    local_search(tend);
    long long cS = sol_cost();
    long long cB = flat_cost(bseq, brend, *bk);
    if (cS < cB) { *bk = flatten(bseq, brend); cB = cS; }
    double tmax = 0.0025 * (double)cB + 1.0, tmin = 0.6;
    int stall = 0;
    while (now() < tend) {
        int sk = flatten(sseq, srend);
        int m = ruin(rem, inrem);
        recreate(rem, m, inrem);
        local_search(tend);
        long long cW = sol_cost();
        double frac = f0 + (f1 - f0) * ((now() - t0) / (tlimit > 1e-9 ? tlimit : 1.0));
        if (frac < 0) frac = 0;
        if (frac > 1) frac = 1;
        double T = tmax * pow(tmin / tmax, frac);
        if (cW < cB) {
            cB = cW;
            *bk = flatten(bseq, brend);
            cS = cW;
            stall = 0;
        } else {
            stall++;
            if (cW <= cS || rndd() < exp(-(double)(cW - cS) / T)) {
                cS = cW;
            } else {
                unflatten(sseq, srend, sk);
                cS = flat_cost(sseq, srend, sk);
            }
            if (stall >= 1000) {
                unflatten(bseq, brend, *bk);
                cS = cB;
                stall = 0;
                for (int i = 0; i < N - 1; i++) {
                    if (rndd() < 0.03) {
                        int j = rnd() % (N - 1);
                        int t = ORD[i];
                        ORD[i] = ORD[j];
                        ORD[j] = t;
                    }
                }
                memset(DLB, 0, n * sizeof(int));
            }
        }
        ensure_empty();
    }
    *ck = flatten(cseq, crend);
    for (int i = 0; i < n; i++) free(RPRE[i]);
    free(RPRE);
    free(RP);
    free(RT); free(RLEN); free(RLOAD); free(RID); free(POS); free(DLB);
    free(ORD); free(TMP); free(NBR); free(rem); free(sseq); free(srend); free(inrem);
}
"""

_LIB = None

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
    """Parallel Clarke-Wright savings with first/last-depot expansion."""
    n = len(demand)
    routes = {c: [c] for c in range(1, n)}
    rid = {c: c for c in range(1, n)}
    load = {c: int(demand[c]) for c in range(1, n)}
    cust = np.arange(1, n)
    iu, ju = np.triu_indices(n - 1, k=1)
    ia, ib = cust[iu], cust[ju]
    sav = dist[0][ia] + dist[0][ib] - dist[ia, ib]
    order = np.argsort(-sav, kind="stable")
    for k_idx in order:
        s = sav[k_idx]
        if s <= 0:
            break
        i = int(ia[k_idx])
        j = int(ib[k_idx])
        ri, rj = rid[i], rid[j]
        if ri == rj or load[ri] + load[rj] > cap:
            continue
        A, B = routes[ri], routes[rj]
        if A[-1] != i:
            if A[0] == i:
                A.reverse()
            else:
                continue
        if B[0] != j:
            if B[-1] == j:
                B.reverse()
            else:
                continue
        A.extend(B)
        for c in B:
            rid[c] = ri
        load[ri] += load[rj]
        del routes[rj], load[rj]
    return list(routes.values())

def _pack(routes, n):
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
    flat = [c for r in routes for c in r]
    if sorted(flat) != list(range(1, n)):
        return False
    return all(sum(int(demand[c]) for c in r) <= cap for r in routes)

def _py_or_opt1(routes, dist, demand, cap, t_end):
    """Or-opt (segments 1,2,3) + intra-route 2-opt, KNN-filtered for speed.
    Mutates routes in place. Only applies strictly-improving moves."""
    n = len(dist)
    if n <= 2:
        return

    max_k = min(15, n - 1)
    knn = np.zeros((n, max_k), dtype=np.int32)
    for c in range(1, n):
        knn[c] = np.argpartition(dist[c, 1:], min(max_k, n - 2))[:max_k] + 1

    loads = [int(sum(int(demand[c]) for c in r)) for r in routes]
    r_map = {}
    for i, r in enumerate(routes):
        for c in r:
            r_map[c] = i

    def rebuild_after(ri, rj):
        loads[ri] = int(sum(int(demand[c]) for c in routes[ri]))
        for c in routes[ri]:
            r_map[c] = ri
        if rj != ri:
            loads[rj] = int(sum(int(demand[c]) for c in routes[rj]))
            for c in routes[rj]:
                r_map[c] = rj

    def try_relocate_S(ri, start, L):
        """Try moving routes[ri][start:start+L] to neighbor routes near segment head."""
        route_i = routes[ri]
        if start + L > len(route_i):
            return False
        a = route_i[start - 1] if start > 0 else 0
        b = route_i[start + L] if start + L < len(route_i) else 0
        seg = route_i[start:start + L]
        seg_load = int(sum(int(demand[c]) for c in seg))
        rem_gain = dist[a, seg[0]] + dist[seg[-1], b] - dist[a, b]

        anchor = seg[0]
        for u in knn[anchor]:
            rj = r_map.get(u)
            if rj is None or rj == ri:
                continue
            if loads[rj] + seg_load > cap:
                continue
            route_j = routes[rj]
            try:
                qu = route_j.index(u)
            except ValueError:
                continue
            # insert after u
            v = route_j[qu + 1] if qu + 1 < len(route_j) else 0
            if v == seg[0]:
                continue
            cost_fwd = dist[u, seg[0]] + dist[seg[-1], v] - dist[u, v]
            if rem_gain - cost_fwd > 0:
                del route_i[start:start + L]
                route_j[qu + 1:qu + 1] = seg
                rebuild_after(ri, rj)
                return True
            # insert before u
            w = route_j[qu - 1] if qu > 0 else 0
            if w == seg[-1]:
                continue
            cost_bwd = dist[w, seg[0]] + dist[seg[-1], u] - dist[w, u]
            if rem_gain - cost_bwd > 0:
                del route_i[start:start + L]
                route_j[qu:qu] = seg
                rebuild_after(ri, rj)
                return True
        return False

    def try_relocate_S_rev(ri, start, L):
        """Try moving reversed segment."""
        route_i = routes[ri]
        if start + L > len(route_i):
            return False
        a = route_i[start - 1] if start > 0 else 0
        b = route_i[start + L] if start + L < len(route_i) else 0
        seg = route_i[start:start + L]
        rseg = seg[::-1]
        seg_load = int(sum(int(demand[c]) for c in seg))
        rem_gain = dist[a, seg[0]] + dist[seg[-1], b] - dist[a, b]

        anchor = rseg[0]
        for u in knn[anchor]:
            rj = r_map.get(u)
            if rj is None or rj == ri:
                continue
            if loads[rj] + seg_load > cap:
                continue
            route_j = routes[rj]
            try:
                qu = route_j.index(u)
            except ValueError:
                continue
            v = route_j[qu + 1] if qu + 1 < len(route_j) else 0
            if v == rseg[0]:
                continue
            cost_fwd = dist[u, rseg[0]] + dist[rseg[-1], v] - dist[u, v]
            if rem_gain - cost_fwd > 0:
                del route_i[start:start + L]
                route_j[qu + 1:qu + 1] = rseg
                rebuild_after(ri, rj)
                return True
            w = route_j[qu - 1] if qu > 0 else 0
            if w == rseg[-1]:
                continue
            cost_bwd = dist[w, rseg[0]] + dist[rseg[-1], u] - dist[w, u]
            if rem_gain - cost_bwd > 0:
                del route_i[start:start + L]
                route_j[qu:qu] = rseg
                rebuild_after(ri, rj)
                return True
        return False

    def try_intra_2opt(ri):
        """Intra-route 2-opt restricted to KNN anchors."""
        route = routes[ri]
        L = len(route)
        if L < 4:
            return False
        for i in range(L - 1):
            anchor = route[i]
            a_prev = route[i - 1] if i > 0 else 0
            a_next = route[i + 1]
            for u in knn[anchor]:
                if u == a_next or u == a_prev or u == anchor:
                    continue
                rj = r_map.get(u)
                if rj != ri:
                    continue
                try:
                    j = route.index(u)
                except ValueError:
                    continue
                if j <= i + 1:
                    continue
                # Reverse route[i+1..j]
                A = route[i]; B = route[i + 1]
                U = route[j]; V = route[j + 1] if j + 1 < L else 0
                gain = dist[A, B] + dist[U, V] - dist[A, U] - dist[B, V]
                if gain > 0:
                    route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                    rebuild_after(ri, ri)
                    return True
        return False

    improved = True
    passes = 0
    while improved and passes < 30:
        if time.monotonic() > t_end:
            break
        improved = False
        passes += 1
        # Or-opt by increasing segment size
        for L in (1, 2, 3):
            if time.monotonic() > t_end:
                break
            scan_changed = True
            while scan_changed:
                if time.monotonic() > t_end:
                    break
                scan_changed = False
                for ri in range(len(routes)):
                    if time.monotonic() > t_end:
                        break
                    if len(routes[ri]) < L:
                        continue
                    for start in range(len(routes[ri]) - L + 1):
                        if time.monotonic() > t_end:
                            break
                        if try_relocate_S(ri, start, L):
                            improved = True
                            scan_changed = True
                            break
                        if time.monotonic() > t_end:
                            break
                        if try_relocate_S_rev(ri, start, L):
                            improved = True
                            scan_changed = True
                            break
                    if scan_changed:
                        break
        # Intra-route 2-opt
        if time.monotonic() > t_end:
            break
        for ri in range(len(routes)):
            if time.monotonic() > t_end:
                break
            if try_intra_2opt(ri):
                improved = True
        # Drop empty routes
        routes[:] = [r for r in routes if r]
        for i, r in enumerate(routes):
            for c in r:
                r_map[c] = i
        loads[:] = [int(sum(int(demand[c]) for c in r)) for r in routes]

def solve(coords, dist, demand, capacity, deadline, compile_c) -> list[list[int]]:
    rng = random.Random(0)
    lib = _lib(compile_c)
    dist = np.ascontiguousarray(dist, dtype=np.int32)
    dem = np.ascontiguousarray(demand, dtype=np.int32)
    n, cap = len(dem), int(capacity)
    if n <= 2:
        return [[1]] if n == 2 else []

    routes = _savings_routes(dist, dem, cap)
    best = [list(r) for r in routes]
    cseq, crend, ck = _pack(routes, n)
    bseq, brend, bk = _pack(routes, n)
    ckv, bkv = ctypes.c_int(ck), ctypes.c_int(bk)
    I = ctypes.POINTER(ctypes.c_int)

    t0 = time.monotonic()
    total = max(deadline - 0.35 - t0, 0.01)
    while True:
        left = deadline - 0.35 - time.monotonic()
        if left <= 0.05:
            break
        sl = min(left, 1.5)
        e0 = time.monotonic() - t0
        try:
            lib.lns(dist.ctypes.data_as(I), dem.ctypes.data_as(I), n, cap,
                    cseq.ctypes.data_as(I), crend.ctypes.data_as(I), ctypes.byref(ckv),
                    bseq.ctypes.data_as(I), brend.ctypes.data_as(I), ctypes.byref(bkv),
                    ctypes.c_double(sl), ctypes.c_double(min(e0 / total, 1.0)),
                    ctypes.c_double(min((e0 + sl) / total, 1.0)),
                    ctypes.c_ulonglong(rng.getrandbits(63) | 1))
            cand = _unpack(bseq, brend, bkv.value)
            if _valid(cand, dem, cap, n):
                t_end = deadline - 0.2
                if time.monotonic() < t_end:
                    _py_or_opt1(cand, dist, dem, cap, t_end)
                if _valid(cand, dem, cap, n):
                    best = cand
                    cseq, crend, ckv = _pack(best, n)
                    ckv = ctypes.c_int(ckv)
        finally:
            cand_f = _unpack(bseq, brend, bkv.value)
            if _valid(cand_f, dem, cap, n):
                c_cost = sum(sum(dist[r[i-1], r[i]] for i in range(len(r))) + dist[0, r[0]] + dist[r[-1], 0] for r in cand_f if r)
                b_cost = sum(sum(dist[r[i-1], r[i]] for i in range(len(r))) + dist[0, r[0]] + dist[r[-1], 0] for r in best if r)
                if c_cost < b_cost:
                    best = cand_f
                    cseq, crend, ckv = _pack(best, n)
                    ckv = ctypes.c_int(ckv)

    # Final polish pass on best with remaining budget
    t_end = deadline - 0.15
    if time.monotonic() < t_end and _valid(best, dem, cap, n):
        cand2 = [list(r) for r in best]
        _py_or_opt1(cand2, dist, dem, cap, t_end)
        if _valid(cand2, dem, cap, n):
            c2_cost = sum(sum(dist[r[i-1], r[i]] for i in range(len(r))) + dist[0, r[0]] + dist[r[-1], 0] for r in cand2 if r)
            b_cost = sum(sum(dist[r[i-1], r[i]] for i in range(len(r))) + dist[0, r[0]] + dist[r[-1], 0] for r in best if r)
            if c2_cost < b_cost:
                best = cand2

    return best