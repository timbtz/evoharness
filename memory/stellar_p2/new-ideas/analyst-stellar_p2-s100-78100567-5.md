# Analyst notes — stellar_p2-s100-78100567 @ 63 candidates
## What the search is doing
The search is entirely gridlocked in a typo-corruption loop around a single hardcoded nfp=3 boundary (`_B3_RCOS`/`_B3_ZSIN`). Over the last 15 attempts (c0035-c0050), every candidate has scored identically at `train 0.6128` or catastrophically failed (`-inf`). The wiki explicitly identifies (`ineffective-approaches/stellar_p2-s100-78100567.md`) that programmatic diffing has persistently corrupted the long float arrays, destroying the exact incumbent matrix and capping the run nearly 0.010 below the proven `s105` official score of 0.6400. 

## Binding problem(s) now
1. **Typo-Corruption Death Spiral:** The hardcoded float matrix is structurally compromised. `c0049` and `c0050` attempted to dynamically pad NAE seeds and bank seeds, but inherited the corrupted matrix as a fallback, scoring 0.6128.
2. **Feasibility-Margin Camping:** Even if the 0.6398 official boundary was successfully reverted to, the wiki's `performance-analysis/feasibility-tolerance-economics.md` proves this score sits at 93% of the feasibility tolerance. At an equal margin (≤ 0.002), it drops to ~0.632, strictly losing to the public leaderboard.
3. **VLF Blindness:** The latest accepted candidate (`c0050`) evaluated a bank seed without verifying it at `low_fidelity`, returning a boundary that scores identically to its parent (untested change) or falling back to a corrupted matrix.

## Decision: pivot — and why
**CONTINUE is a guaranteed loss** (the incumbent matrix is corrupted). **REVIVE is useless** (all prior solutions in this branch are corrupted descendants of the public #1 seed). **PIVOT** is the only mathematically grounded escape. 

The campaign's core failure—identified heavily across the wiki (`provenance-and-independence.md`, `low-nfp-nae.md`)—is that every scoring boundary is a <0.5% perturbation of the public #1 submission, sitting at the absolute aspect-ratio limit. The only untested multiplicative lever for the target objective ($L \propto A/N_{fp}$) is dropping the field-period count to **nfp=2**. If the QI constraint holds at lower nfp, $L$ jumps categorically; if it fails, we cleanly fall back to an uncorrupted `fm.seed_bank(6)` contraction. I am replacing the entire corrupted hardcoded approach with a dynamically-generated nfp=2 NAE basin search paired with a pristine bank-seed fallback.

## Proposal (the ONE candidate you inject: idea, mechanism, expected effect)
**Idea:** A complete pivot to an **nfp=2 `seed_nae` basin search** combined with the proven R/Z contraction mechanism, guaranteeing a mathematically pristine boundary free of float-corruption and public-seed proximity.
**Mechanism:** 
1. Definitively discard the typo-ridden `_B3_RCOS` matrices. 
2. Generate 6 distinct nfp=2 NAE seeds targeting varying aspect ratios (9.0-10.0).
3. Apply the proven two-stage R/Z split depth-contraction (`cr=0.50`, `cz=0.70`) to push the aspect ratio to its tolerance wall.
4. Safely batch these alongside 3 pristine `fm.seed_bank(6)` contractions as a non-regressing fallback floor.
5. Evaluate in a single `eval_many` batch and select via the `honest_score` + novelty penalty.
**Expected Effect:** The nfp=2 candidates either break the QI barrier and yield a massive structural leap in $L$ (uncorrupted and highly novel), or safely land at the `seed_bank` floor of ~0.620 train (avoiding the 0.6128 corruption trap).

## Decision log (alternatives considered and rejected, with reasons)
- **CONTINUE (fix the hardcoded floats):** Rejected. The wiki tracks 15+ consecutive failures to repair the exact decimal arrays programmatically. Any further manual diffing is structurally doomed.
- **REVIVE (pure unmodified `fm.seed_bank(6)`):** Rejected as a primary objective. Yields a safe ~0.620 floor, but fails the novelty requirement (bank_cos will heavily penalize it). Used only as an emergency fallback in the batch.
- **PIVOT (Cross-bank mode grafting):** Rejected. Wiki marks this exhausted (`mode-grafting-and-blends.md`); blending Fourier modes across different basins destroys the delicate magnetic surface QI coupling.
