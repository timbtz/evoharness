"""C matmul micro-kernel task (AlphaDev/KernelBench spirit): evolve C source,
score = measured speedup vs the naive kernel timed in the same binary
(same-binary relative timing cancels machine-load noise to first order)."""
from __future__ import annotations

from pathlib import Path

from core.candidate import EvalResult
from core.sandbox import run_c
from tasks.common import run_harness

_SPLITS = {"train": [192, 256], "val": [224], "public": [320, 384], "private": [128, 512]}

_TEMPLATE = r'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

__CODE__

static unsigned long long rs;
static double rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (double)(rs >> 11) / 9007199254740992.0; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec; }

static void naive(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}

static double bench(void (*f)(const double*, const double*, double*, int),
                    const double* A, const double* B, double* C, int n) {
    double t0 = now(); f(A, B, C, n); double warm = now() - t0;
    int reps = warm < 0.04 ? (int)(0.04 / warm) + 1 : 1;
    double best = 1e30;
    for (int b = 0; b < 2; b++) {
        t0 = now();
        for (int r = 0; r < reps; r++) f(A, B, C, n);
        double t = (now() - t0) / reps;
        if (t < best) best = t;
    }
    return best;
}

int main(void) {
    int sizes[] = {__SIZES__};
    int ns = sizeof(sizes) / sizeof(int);
    double sp[16], gf[16], logsum = 0.0;
    for (int s = 0; s < ns; s++) {
        int n = sizes[s];
        double *A = malloc(sizeof(double)*n*n), *B = malloc(sizeof(double)*n*n);
        double *Cr = malloc(sizeof(double)*n*n), *Cc = malloc(sizeof(double)*n*n);
        rs = 0x9E3779B97F4A7C15ULL + (unsigned long long)s * 1234567ULL;
        for (int i = 0; i < n*n; i++) { A[i] = rnd() - 0.5; B[i] = rnd() - 0.5; }
        naive(A, B, Cr, n);
        matmul(A, B, Cc, n);
        double diff = 0.0;
        for (int i = 0; i < n*n; i++) { double d = fabs(Cc[i] - Cr[i]); if (d > diff) diff = d; }
        if (diff > 1e-7 * n || diff != diff) {
            printf("{\"error\": \"wrong result at n=%d: max abs diff %.3g\"}\n", n, diff);
            return 0;
        }
        double tn = bench(naive, A, B, Cr, n), tc = bench(matmul, A, B, Cc, n);
        sp[s] = tn / tc; gf[s] = 2.0 * n * n * (double)n / tc / 1e9;
        logsum += log(sp[s]);
        free(A); free(B); free(Cr); free(Cc);
    }
    printf("{\"score\": %.4f, \"metrics\": {\"speedup_gm\": %.4f, \"speedups\": [",
           exp(logsum / ns), exp(logsum / ns));
    for (int s = 0; s < ns; s++) printf("%s%.3f", s ? ", " : "", sp[s]);
    printf("], \"gflops\": [");
    for (int s = 0; s < ns; s++) printf("%s%.2f", s ? ", " : "", gf[s]);
    printf("], \"sizes\": [");
    for (int s = 0; s < ns; s++) printf("%s%d", s ? ", " : "", sizes[s]);
    printf("]}}\n");
    return 0;
}
'''

_SEED = """void matmul(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[i*n+k] * B[k*n+j];
            C[i*n+j] = s;
        }
}"""


class _MatmulCTask:
    name = "matmul_c"
    wiki_dir = Path(__file__).parent / "wiki"
    description = """\
Optimize a C matrix-multiply micro-kernel for SPEED on one x86-64 core
(AMD EPYC-Rome, gcc -O3 -march=native, single-threaded — OpenMP is not enabled).
Write exactly this function in C (row-major n*n matrices, C = A*B):

    void matmul(const double* A, const double* B, double* C, int n)

Provide ONLY this function plus any helper functions/#includes it needs
(<immintrin.h> intrinsics allowed). Do NOT define main(). n is a multiple
of 32 in [128, 512]; C may contain garbage on entry and must be fully written.
Correctness is checked against a naive reference (max abs diff <= 1e-7*n),
then both kernels are timed in the same binary; score = geometric-mean
speedup over the naive i-j-k loop (seed scores ~1.0; loop reordering to
i-k-j typically gives 2-4x, cache blocking + unrolling more). Reassociating
the k-sum is fine (tolerance allows it). Output one fenced code block."""

    def seed_code(self) -> str:
        return _SEED

    def evaluate(self, code: str, split: str) -> EvalResult:
        sizes = _SPLITS.get(split)
        if sizes is None:
            return EvalResult(float("-inf"), error=f"unknown split: {split}")
        src = _TEMPLATE.replace("__CODE__", code).replace(
            "__SIZES__", ", ".join(map(str, sizes)))
        return run_harness(src, timeout=90.0 if split == "private" else 60.0, runner=run_c)

    def render(self, code: str, result: EvalResult) -> dict:
        m = result.metrics or {}
        return {"kind": "bars", "labels": m.get("sizes", []),
                "values": m.get("speedups", []), "title": "speedup vs naive",
                "speedup_gm": m.get("speedup_gm")}


TASK = _MatmulCTask()
