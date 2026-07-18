# Benchmarks

## Instance table (best-known cost, CVRPLIB rounded-euclidean convention;
source galgos.inf.puc-rio.br/cvrplib + PyVRP/Instances, verified 2026-07-18)

| split   | instance    | n   | BKS   |
|---------|-------------|-----|-------|
| train   | A-n32-k5    | 32  | 784   |
| train   | A-n45-k6    | 45  | 944   |
| train   | A-n60-k9    | 60  | 1354  |
| train   | B-n50-k7    | 50  | 741   |
| train   | P-n55-k10   | 55  | 694   |
| val     | A-n37-k6    | 37  | 949   |
| val     | B-n45-k5    | 45  | 751   |
| val     | P-n65-k10   | 65  | 792   |
| public  | X-n101-k25  | 101 | 27591 |
| public  | X-n110-k13  | 110 | 14971 |
| public  | X-n125-k30  | 125 | 55539 |
| hidden  | larger X    | up to ~300 | BKS (mostly not proven optimal) |

A/B/P and the smaller X above are proven optima; larger private-split X are
best-known, not all proven optimal.

## Set structure
- **A**: random-uniform customer coordinates, moderate capacity.
- **B**: clustered customers — sweep/angular construction underperforms here
  (construction.md); savings/regret handle clusters better.
- **P**: capacities modified tighter relative to demand => more, smaller
  routes; regret-k insertion and 2-opt* matter most (fewer feasible slots).
- **X** (public + hidden): generated instances, mixed depot positioning and
  demand distributions, much larger n (101-303).

## Gap expectations (gap% = 100*(cost-BKS)/BKS)
| search level                          | typical gap |
|----------------------------------------|-------------|
| plain savings only                     | 8-12%       |
| + 2-opt / relocate                     | 5-8%        |
| + swap / Or-opt / 2-opt*               | 3-5%        |
| LNS / ruin-recreate, well tuned        | 1-3%        |
| state-of-the-art (HGS, far larger budgets) | <1%     |

## Anytime lever
- n<=65 (train/val, 3s budget): local search converges to a local optimum in
  well under a second — the remaining budget is spent entirely on perturbation
  quality. Score differentiates on the ANYTIME POLICY (LNS round count, ruin
  selection), not kernel micro-optimization; see anytime.md.
- n=101-303 (public 5s / private 6s): initial convergence itself consumes a
  meaningful slice of the budget, so raw kernel speed (candidate lists,
  don't-look bits — local-search.md) starts to directly gate how many
  perturbation rounds fit alongside policy quality.
