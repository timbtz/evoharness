# B4 Truncation, Same-nfp Homotopy, and Bank-Seed Structural Escapes
Applying structural truncation, homotopy blends, or direct envelope shifts specifically on Bank #4 (B4) RisoLiao or top bank seeds fails to beat the nfp=3 floor, often timing out due to the high eval cost of mpol=8 nfp=4 boundaries.
## How it was tried
- `stellar_p2-s105-72881323` c0076 (train 0.6255, val 0.6368): Attempted structural truncation of B4 RisoLiao (mpol=5) + aspect shifts alongside the nfp=3 floor. Collapsed to the fallback floor.
- `stellar_p2-s105-72881323` c0077r1 (ERR): Applied a 15-candidate truncation × contraction grid on B4. Timed out at 720s.
- `stellar_p2-s105-72881323` c0078r1 (train 0.6255, val 0.6368): Truncated B4 to mpol=5 + R/Z contraction sweep. Regressed to the nfp=3 fallback floor.
- `stellar_p2-s105-72881323` c0080 (train 0.6246, val 0.6360): Replaced homotopy with truncated Bank#4 RisoLiao (mpol=6/ntor=2) contraction sweep. Regressed score and wasted budget.
- `stellar_p2-s105-72881323` c0084 (train 0.6255, val 0.6368): Loaded top bank seeds via `fm.seed_bank(i)` and applied structural truncation + envelope modulation. Collapsed straight to the fallback floor.
## Why it failed
Writers predicted that B4's zero violation would provide massive headroom, or that truncation would reduce eval cost enough to sweep. The code did exactly this, but standalone B4 sweeps simply starve the budget (timeouts) and mid-grid refinements fail to overcome the structural cap of ~0.6330 official for this basin family. The bank seed is structurally Pareto-blocked; any structural damping or truncation of its modes disrupts the delicate spectral balance.
## Verdict
exhausted — Do not run standalone sweeps or structural truncations on B4 or top bank seeds.
