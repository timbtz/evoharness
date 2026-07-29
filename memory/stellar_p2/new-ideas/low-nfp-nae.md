# Low-nfp (nfp=2) NAE Basin Pivot
Pivoting to a near-axis (NAE) nfp=2 seed to exploit the theoretical scaling L ∝ A/Nfp for a category jump in L.

## Proposal
- **Idea:** Seed a fresh nfp=2 QI basin via `fm.seed_nae()`. Since aspect ratio (A) is pinned at the ~10.10 wall on the nfp=3 B3 basin, reducing nfp from 3 to 2 theoretically provides a ~1.5x multiplier on L (12.5 → ~18-19) if QI can be maintained.
- **Selection:** Batch A (one `eval_many`) = the c0034-frontier floor (guaranteeing ≥0.6269) + ~9 free `fm.seed_nae` seeds at **nfp=2** (aspect 9–10, iota≥0.55, elong≤4.5, mirror≤0.15, modes 4×4). If any low-nfp seed is competitive, Batch B applies the proven R/Z contraction to trim aspect.
- **Expected effect:** If a tuned nfp=2 seed lands log10_qi ≤ -4 at aspect ≤ 10, L jumps to ~16-19 (score ≫ 0.64) in a novel basin; otherwise it returns the ~0.627 c0034 floor safely.
- **Risk:** nfp=2 worsens omnigenity and the QI residual is already thin (log10_qi -4.0025). Whether QI holds is the single empirical question. 

## Verdict
promising but barely tested — This is the only mathematically grounded lever left for a category jump. It was probed in `stellar_p2-s100-78100567` c0001 and c0002, but both implementations fundamentally failed to construct the NAE sweep properly. In c0001, the writer botched the `eval_many` batch format; in c0002, the NAE loop was completely dead code (omitted from the `eval_many` batch) and the candidate simply evaluated the shallow contraction grid, scoring train 0.6137. The nfp=2 basin remains genuinely untested.
