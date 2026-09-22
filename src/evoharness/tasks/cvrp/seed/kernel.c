/* kernel.c — CVRP granular local search + LNS engine (the seed's C half).
 *
 * Exported interface (ctypes; int = int32, row-major dist, node 0 = depot):
 *
 *   void lns(const int *dist, const int *demand, int n, int cap,
 *            int *cur_seq, int *cur_rend, int *cur_k,
 *            int *best_seq, int *best_rend, int *best_k,
 *            double tlimit, double frac0, double frac1,
 *            unsigned long long seed);
 *
 *   cur_*  : current LNS solution, in/out. seq = customers of all routes
 *            concatenated, rend[r] = exclusive end of route r, *_k = #routes.
 *            Buffers must hold n ints each.
 *   best_* : best solution found so far, in/out (only overwritten on improve).
 *   tlimit : wall-clock seconds this call may use (checked inside all loops).
 *   frac0/1: global elapsed-time fraction at slice start/end so the annealing
 *            schedule cools continuously across time-sliced calls.
 *   seed   : rng seed; caller passes a deterministic sequence.
 *
 * Standalone build: gcc -O3 -march=native -shared -fPIC -o /tmp/k.so kernel.c
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* xorshift64* — small deterministic rng */
static unsigned long long RS;
static unsigned int rnd(void) {
    RS ^= RS >> 12; RS ^= RS << 25; RS ^= RS >> 27;
    return (unsigned int)((RS * 2685821657736338717ULL) >> 32);
}
static double rndd(void) { return rnd() * (1.0 / 4294967296.0); }

/* Problem + working solution. Routes live in an n x n array (row per route),
 * with per-customer route id RID and position POS so every move is O(1) to
 * locate. DLB = don't-look bits (Bentley); NBR = K-nearest-neighbor lists. */
static const int *DIST, *DEM;
static int N, CAP, K, R;
static int *RT, *RLEN, *RLOAD, *RID, *POS, *NBR, *DLB, *ORD, *TMP;
#define D(a, b) DIST[(long)(a) * N + (b)]
#define AT(r, i) RT[(r) * N + (i)]

static int prvn(int r, int i) { return i > 0 ? AT(r, i - 1) : 0; }
static int nxtn(int r, int i) { return i < RLEN[r] - 1 ? AT(r, i + 1) : 0; }
static void wake(int c) { if (c) DLB[c] = 0; }

static void reload(int r) {              /* rebuild POS/RID/RLOAD of a route */
    int ld = 0;
    for (int i = 0; i < RLEN[r]; i++) {
        int c = AT(r, i); POS[c] = i; RID[c] = r; ld += DEM[c];
    }
    RLOAD[r] = ld;
}

/* remove L customers starting at position p of route r into buf */
static void seg_out(int r, int p, int L, int *buf) {
    wake(prvn(r, p)); wake(nxtn(r, p + L - 1));
    memcpy(buf, &AT(r, p), (size_t)L * sizeof(int));
    for (int i = 0; i < L; i++) RLOAD[r] -= DEM[buf[i]];
    memmove(&AT(r, p), &AT(r, p + L), (size_t)(RLEN[r] - p - L) * sizeof(int));
    RLEN[r] -= L;
    for (int i = p; i < RLEN[r]; i++) POS[AT(r, i)] = i;
}

/* insert buf[0..L) into route r before position q */
static void seg_in(int r, int q, int L, const int *buf) {
    wake(q > 0 ? AT(r, q - 1) : 0); wake(q < RLEN[r] ? AT(r, q) : 0);
    memmove(&AT(r, q + L), &AT(r, q), (size_t)(RLEN[r] - q) * sizeof(int));
    memcpy(&AT(r, q), buf, (size_t)L * sizeof(int));
    RLEN[r] += L;
    for (int i = 0; i < L; i++) { RLOAD[r] += DEM[buf[i]]; RID[buf[i]] = r; wake(buf[i]); }
    for (int i = q; i < RLEN[r]; i++) POS[AT(r, i)] = i;
}

static void rev_seg(int r, int i, int j) {   /* intra-route 2-opt reversal */
    wake(prvn(r, i)); wake(AT(r, i)); wake(AT(r, j)); wake(nxtn(r, j));
    int *a = &AT(r, 0);
    while (i < j) {
        int t = a[i]; a[i] = a[j]; a[j] = t;
        POS[a[i]] = i; POS[a[j]] = j; i++; j--;
    }
}

static int prefload(int r, int p) {          /* load of route prefix [0..p] */
    int s = 0;
    for (int i = 0; i <= p; i++) s += DEM[AT(r, i)];
    return s;
}

/* 2-opt*: exchange the tail after position p of r with the tail after q of
 * r2 (p or q may be -1 = whole route is the tail). Toth & Vigo. */
static void tails(int r, int p, int r2, int q) {
    int L1 = RLEN[r] - p - 1, L2 = RLEN[r2] - q - 1;
    memcpy(TMP, &AT(r, p + 1), (size_t)L1 * sizeof(int));
    memcpy(&AT(r, p + 1), &AT(r2, q + 1), (size_t)L2 * sizeof(int));
    RLEN[r] = p + 1 + L2;
    memcpy(&AT(r2, q + 1), TMP, (size_t)L1 * sizeof(int));
    RLEN[r2] = q + 1 + L1;
    reload(r); reload(r2);
    if (p >= 0) wake(AT(r, p));
    if (p + 1 < RLEN[r]) wake(AT(r, p + 1));
    if (q >= 0) wake(AT(r2, q));
    if (q + 1 < RLEN[r2]) wake(AT(r2, q + 1));
}

static void build_knn(void) {                /* K nearest customers of each c */
    for (int c = 1; c < N; c++) {
        for (int u = 1; u < N; u++) TMP[u] = 0;
        for (int t = 0; t < K; t++) {
            int bu = -1, bd = 0;
            for (int u = 1; u < N; u++)
                if (u != c && !TMP[u] && (bu < 0 || D(c, u) < bd)) { bu = u; bd = D(c, u); }
            NBR[c * K + t] = bu; TMP[bu] = 1;
        }
    }
}

static long long sol_cost(void) {
    long long s = 0;
    for (int r = 0; r < R; r++) {
        int pr = 0;
        for (int i = 0; i < RLEN[r]; i++) { s += D(pr, AT(r, i)); pr = AT(r, i); }
        s += D(pr, 0);
    }
    return s;
}

static int flatten(int *seq, int *rend) {    /* working -> flat, drops empties */
    int k = 0, off = 0;
    for (int r = 0; r < R; r++) {
        if (!RLEN[r]) continue;
        memcpy(seq + off, &AT(r, 0), (size_t)RLEN[r] * sizeof(int));
        off += RLEN[r]; rend[k++] = off;
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
            reload(R); R++;
        }
        s = e;
    }
    RLEN[R] = 0; RLOAD[R] = 0; R++;   /* keep one empty route: vehicle count may grow */
}

static long long flat_cost(const int *seq, const int *rend, int k) {
    long long s = 0;
    for (int r = 0, st = 0; r < k; r++) {
        int pr = 0;
        for (int i = st; i < rend[r]; i++) { s += D(pr, seq[i]); pr = seq[i]; }
        s += D(pr, 0); st = rend[r];
    }
    return s;
}

static void ensure_empty(void) {  /* compact away extra empties, keep exactly one */
    int w = 0;
    for (int r = 0; r < R; r++)
        if (RLEN[r]) {
            if (w != r) {
                memcpy(&AT(w, 0), &AT(r, 0), (size_t)RLEN[r] * sizeof(int));
                RLEN[w] = RLEN[r]; reload(w);
            }
            w++;
        }
    RLEN[w] = 0; RLOAD[w] = 0; R = w + 1;
}

/* First-improvement granular neighborhood of customer c: only arcs (c,u) with
 * u among c's K nearest neighbors are considered (granular search, Toth &
 * Vigo 2003). Move set follows the classical canon (cf. Vidal et al. 2012):
 * relocate, Or-opt segments of 2-3 both orientations (Or 1976), swap,
 * intra-route 2-opt, and inter-route 2-opt* (both tail pairings). */
static int try_moves(int c) {
    int b1[3];
    int r = RID[c], p = POS[c];
    int a = prvn(r, p), x = nxtn(r, p);
    int x2 = x ? nxtn(r, p + 1) : 0, x3 = x2 ? nxtn(r, p + 2) : 0;
    int rem1 = D(a, c) + D(c, x) - D(a, x);              /* gain of ejecting c */
    int rem2 = x ? D(a, c) + D(x, x2) - D(a, x2) : 0;    /* ... segment [c,x] */
    int rem3 = x2 ? D(a, c) + D(x2, x3) - D(a, x3) : 0;  /* ... [c,x,x2] */
    int q1 = DEM[c], q2 = q1 + (x ? DEM[x] : 0), q3 = q2 + (x2 ? DEM[x2] : 0);
    for (int t = 0; t < K; t++) {
        int u = NBR[c * K + t];
        int r2 = RID[u], q = POS[u];
        int w = prvn(r2, q), v = nxtn(r2, q);
        int okc = r2 == r || RLOAD[r2] + q1 <= CAP;
        /* relocate c after u */
        if (okc && u != a) {
            int g = rem1 - (D(u, c) + D(c, v) - D(u, v));
            if (g > 0) { seg_out(r, p, 1, b1); seg_in(r2, POS[u] + 1, 1, b1); return 1; }
        }
        /* relocate c before u */
        if (okc && w != c) {
            int g = rem1 - (D(w, c) + D(c, u) - D(w, u));
            if (g > 0) { seg_out(r, p, 1, b1); seg_in(r2, POS[u], 1, b1); return 1; }
        }
        /* Or-opt: move segment [c,x] after u, forward or reversed */
        if (x && u != x && u != a && (r2 == r || RLOAD[r2] + q2 <= CAP)) {
            int gf = rem2 - (D(u, c) + D(x, v) - D(u, v));
            int gr = rem2 - (D(u, x) + D(c, v) - D(u, v));
            if (gf > 0 || gr > 0) {
                seg_out(r, p, 2, b1);
                if (gr > gf) { int tt = b1[0]; b1[0] = b1[1]; b1[1] = tt; }
                seg_in(r2, POS[u] + 1, 2, b1); return 1;
            }
        }
        /* Or-opt: move segment [c,x,x2] after u, forward or reversed */
        if (x2 && u != x && u != x2 && u != a && (r2 == r || RLOAD[r2] + q3 <= CAP)) {
            int gf = rem3 - (D(u, c) + D(x2, v) - D(u, v));
            int gr = rem3 - (D(u, x2) + D(c, v) - D(u, v));
            if (gf > 0 || gr > 0) {
                seg_out(r, p, 3, b1);
                if (gr > gf) { int tt = b1[0]; b1[0] = b1[2]; b1[2] = tt; }
                seg_in(r2, POS[u] + 1, 3, b1); return 1;
            }
        }
        /* swap c <-> u (adjacent same-route pairs are 2-opt/Or-opt territory) */
        if ((r2 != r && RLOAD[r] - q1 + DEM[u] <= CAP && RLOAD[r2] - DEM[u] + q1 <= CAP)
            || (r2 == r && q != p - 1 && q != p + 1)) {
            int g = D(a, c) + D(c, x) + D(w, u) + D(u, v)
                  - (D(a, u) + D(u, x) + D(w, c) + D(c, v));
            if (g > 0) {
                AT(r, p) = u; AT(r2, q) = c;
                POS[u] = p; POS[c] = q; RID[u] = r; RID[c] = r2;
                if (r2 != r) { RLOAD[r] += DEM[u] - q1; RLOAD[r2] += q1 - DEM[u]; }
                wake(a); wake(x); wake(w); wake(v); wake(c); wake(u);
                return 1;
            }
        }
        if (r2 == r) {
            /* intra-route 2-opt: drop edges (i,i+1) and (j,succ j), reverse */
            int i = p < q ? p : q, j = p < q ? q : p;
            int A = AT(r, i), B = AT(r, i + 1), E = nxtn(r, j);
            int g = D(A, B) + D(AT(r, j), E) - D(A, AT(r, j)) - D(B, E);
            if (g > 0) { rev_seg(r, i + 1, j); return 1; }
        } else {
            /* 2-opt* pairing 1: tail after c <-> tail after u */
            int g = D(c, x) + D(u, v) - D(c, v) - D(u, x);
            if (g > 0) {
                int plc = prefload(r, p), plu = prefload(r2, q);
                if (plc + RLOAD[r2] - plu <= CAP && plu + RLOAD[r] - plc <= CAP) {
                    tails(r, p, r2, q); return 1;
                }
            }
            /* 2-opt* pairing 2: tail after c <-> tail starting at u (edge c-u) */
            g = D(c, x) + D(w, u) - D(c, u) - D(w, x);
            if (g > 0) {
                int plc = prefload(r, p), plw = prefload(r2, q) - DEM[u];
                if (plc + RLOAD[r2] - plw <= CAP && plw + RLOAD[r] - plc <= CAP) {
                    tails(r, p, r2, q - 1); return 1;
                }
            }
        }
    }
    /* open a fresh route when c's detour exceeds two depot legs */
    if (rem1 > 2 * D(0, c))
        for (int r2 = 0; r2 < R; r2++)
            if (!RLEN[r2]) { b1[0] = c; seg_out(r, p, 1, b1); seg_in(r2, 0, 1, b1); return 1; }
    return 0;
}

/* Sweep customers (shuffled order) until no move fires. Integer distances
 * mean every applied move gains >= 1, so convergence is finite. */
static void local_search(double tend) {
    for (int i = N - 2; i > 0; i--) {
        int j = rnd() % (i + 1);
        int t = ORD[i]; ORD[i] = ORD[j]; ORD[j] = t;
    }
    int improved = 1;
    while (improved) {
        improved = 0;
        for (int t = 0; t < N - 1; t++) {
            if ((t & 15) == 0 && now() > tend) return;
            int c = ORD[t];
            if (DLB[c]) continue;
            if (try_moves(c)) improved = 1; else DLB[c] = 1;
        }
    }
}

/* Ruin: Shaw-style related removal (Shaw 1998; relatedness = distance, rank-
 * biased) alternating with random contiguous segment removal. ~8-18% of
 * customers. Returns count; removed ids in rem[], flagged in inrem[]. */
static int ruin(int *rem, char *inrem) {
    int nc = N - 1, cnt = 0;
    int m = (int)((0.08 + 0.10 * rndd()) * nc);
    if (m < 5) m = 5;
    if (m > 60) m = 60;
    if (m > nc - 2) m = nc - 2;
    if (m < 1) m = 1;
    memset(inrem, 0, N);
    if (rndd() < 0.5) {
        rem[cnt] = 1 + rnd() % nc; inrem[rem[cnt]] = 1; cnt++;
        while (cnt < m) {
            int c = rem[rnd() % cnt], nc2 = 0, pick;
            for (int t = 0; t < K; t++) {
                int u = NBR[c * K + t];
                if (!inrem[u]) TMP[nc2++] = u;
            }
            if (nc2) pick = TMP[(int)(pow(rndd(), 3.0) * nc2)];
            else do pick = 1 + rnd() % nc; while (inrem[pick]);
            inrem[pick] = 1; rem[cnt++] = pick;
        }
        for (int i = 0; i < cnt; i++) seg_out(RID[rem[i]], POS[rem[i]], 1, TMP);
    } else {
        while (cnt < m) {
            int r = rnd() % R;
            if (!RLEN[r]) continue;
            int maxl = RLEN[r] < m - cnt ? RLEN[r] : m - cnt;
            int L = 1 + rnd() % maxl, st = rnd() % (RLEN[r] - L + 1);
            for (int i = 0; i < L; i++) { int u = AT(r, st + i); rem[cnt++] = u; inrem[u] = 1; }
            seg_out(r, st, L, TMP);
        }
    }
    return cnt;
}

/* Recreate: regret-2 insertion. Each round, insert the customer with the
 * largest gap between its best and second-best route (infinite if only one
 * route fits), at its cheapest position. An empty route is always available,
 * so insertion cannot fail and the fleet can grow when it must. */
static void recreate(int *rem, int m, char *inrem) {
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
                    int nx = qq < RLEN[r] ? AT(r, qq) : 0;
                    long long inc = D(pr, c) + D(c, nx) - D(pr, nx);
                    if (inc < rb) { rb = inc; rq = qq; }
                    pr = nx;
                }
                if (rb < b1) { b2 = b1; b1 = rb; r1 = r; o1 = rq; }
                else if (rb < b2) b2 = rb;
            }
            long long reg = b2 - b1;
            if (reg > bg || (reg == bg && b1 < bb)) { bg = reg; bb = b1; bc = c; br = r1; bq = o1; }
        }
        seg_in(br, bq, 1, &bc);
        inrem[bc] = 0;
        int has = 0;
        for (int r = 0; r < R; r++) if (!RLEN[r]) { has = 1; break; }
        if (!has && R < N) { RLEN[R] = 0; RLOAD[R] = 0; R++; }
    }
}

/* LNS outer loop: ruin & recreate + granular LS, SA-lite acceptance cooling
 * over global time (record-to-record flavour), restart from best on long
 * stalls. cur_* continues across slices; best_* only improves. */
void lns(const int *dist, const int *demand, int n, int cap,
         int *cseq, int *crend, int *ck,
         int *bseq, int *brend, int *bk,
         double tlimit, double f0, double f1, unsigned long long seed) {
    double t0 = now(), tend = t0 + tlimit;
    DIST = dist; DEM = demand; N = n; CAP = cap;
    K = n - 2 < 24 ? n - 2 : 24;
    if (K < 1) K = 1;
    RT = malloc((size_t)n * n * sizeof(int));
    RLEN = malloc(n * sizeof(int)); RLOAD = malloc(n * sizeof(int));
    RID = malloc(n * sizeof(int)); POS = malloc(n * sizeof(int));
    DLB = malloc(n * sizeof(int)); ORD = malloc(n * sizeof(int));
    TMP = malloc(n * sizeof(int)); NBR = malloc((size_t)n * K * sizeof(int));
    int *rem = malloc(n * sizeof(int));
    int *sseq = malloc(n * sizeof(int)), *srend = malloc(n * sizeof(int));
    char *inrem = malloc(n);
    RS = seed | 1;
    build_knn();
    for (int i = 0; i < N - 1; i++) ORD[i] = i + 1;
    memset(DLB, 0, n * sizeof(int));
    unflatten(cseq, crend, *ck);
    local_search(tend);
    long long cS = sol_cost(), cB = flat_cost(bseq, brend, *bk);
    if (cS < cB) { *bk = flatten(bseq, brend); cB = cS; }
    double tmax = 0.003 * (double)cB + 1.0, tmin = 0.7;
    int stall = 0;
    while (now() < tend) {
        int sk = flatten(sseq, srend);       /* snapshot of current */
        int m = ruin(rem, inrem);
        recreate(rem, m, inrem);
        local_search(tend);
        long long cW = sol_cost();
        double frac = f0 + (f1 - f0) * ((now() - t0) / (tlimit > 1e-9 ? tlimit : 1.0));
        if (frac < 0) frac = 0;
        if (frac > 1) frac = 1;
        double T = tmax * pow(tmin / tmax, frac);
        if (cW < cB) {
            cB = cW; *bk = flatten(bseq, brend); cS = cW; stall = 0;
        } else {
            stall++;
            if (cW <= cS || rndd() < exp(-(double)(cW - cS) / T)) cS = cW;
            else unflatten(sseq, srend, sk);           /* reject: roll back */
            if (stall >= 1200) {                       /* restart from best */
                unflatten(bseq, brend, *bk); cS = cB; stall = 0;
                memset(DLB, 0, n * sizeof(int));
            }
        }
        ensure_empty();
    }
    *ck = flatten(cseq, crend);
    free(RT); free(RLEN); free(RLOAD); free(RID); free(POS); free(DLB);
    free(ORD); free(TMP); free(NBR); free(rem); free(sseq); free(srend); free(inrem);
}
