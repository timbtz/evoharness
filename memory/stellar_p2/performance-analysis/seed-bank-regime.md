# The seed-bank regime: the game is now beating 0.636
fm.seed_bank exposes 12 public leaderboard P2 submissions (official 0.40-0.636,
all nfp=3); the novelty bar is now BEATING the bank max, not reaching
feasibility.

## Measured (2026-07-24, vlf = train fidelity)
- Top-7 bank seeds hold 0.60-0.62 at vlf (official 0.61-0.64): resolution gap
  only 0.01-0.02 — vlf polish signals are trustworthy here.
- 3 seeds (official 0.434-0.537) flip INFEASIBLE at vlf (violations
  0.0106-0.0122 vs 0.01) — tolerance camping in the wild; polishing those
  first requires re-entering feasibility.
- COST: high-mode bank boundaries take 12-27 s/eval at vlf ((8,15) and
  (11,21) matrices) vs 1.4 s for mp=1. A 480 s CPU budget buys only ~25-35
  bank-boundary evals. Eval frugality IS the game.
- L headroom: bank best L~12.3 at vlf (score 0.61); score 1.0 needs L=20.
  Nobody knows how much headroom this basin has — leaderboard #1 stopped at
  0.636.

## Rules (honesty, non-negotiable)
Anything seeded from the bank is "refined from public submissions", never
"from scratch". Returning a bank boundary unimproved or trivially perturbed
is worthless AND unsubmittable (export refuses near-copies < 1e-3).

## Open directions for this regime
1. Recombination: blend/graft coefficients between the 7 good seeds (they are
   DIFFERENT local optima — davidkh (8,15) vs phanerozoic (11,21) vs RisoLiao).
2. Feasibility-margin-aware polish: maximize L subject to violations staying
   < ~0.007 at vlf (private tolerance is tighter in practice).
3. Repair the 3 tolerance-campers back into feasibility (+0.43-0.54 archive
   entries if it works).
4. Mode truncation of (11,21) seeds -> cheaper evals, maybe same score.
