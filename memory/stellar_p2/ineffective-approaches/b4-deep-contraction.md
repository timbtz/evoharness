# Dedicated Deep Contraction on Bank #4 (RisoLiao)
Applying deep R/Z split differential contractions specifically to Bank #4 (B4) as a standalone strategy plateaus immediately and risks critical timeouts due to high eval costs.
## How it was tried
- `stellar_p2-s105-72881323` c0064 (ERR): Evaluated a 9-candidate deep contraction sweep (base_c -5.0e-3 to +1.0e-3) directly on B4. Timed out after 720s because nfp=4 evaluations cost 12-27s each.
- `stellar_p2-s105-72881323` c0065r2 (train 0.6211): Reduced portfolio (5 candidates) applying R/Z split contraction with m-shifts directly to B4. Regressed to the B1 fallback floor.
- `stellar_p2-s105-72881323` c0066 (train 0.6237): Swept a finer 5-point R/Z curvature split (0.4, 0.8) on B1 and B3. Failed to improve over the exact 0.5/0.7 off-diagonal split, returning the fallback floor.
## Why it failed
The writer predicted that B4's zero violation (vs B1's 0.0015 log10_qi wall) would provide ~6x more feasibility headroom to absorb aspect-ratio reduction, pushing L upward. The code applied this directly but failed: standalone B4 sweeps starve the budget (timeouts) and standalone mid-grid refinements simply confirm the (0.5, 0.7) off-diagonal split is a strict optimum.
## Verdict
exhausted — Do not run standalone deep sweeps on B4. B4 is only viable when batched efficiently within a tri-basin portfolio.
