# The fidelity dial and where the CPU goes
Forward-model cost spans ~175x; the score you optimize on train is a cheap proxy and only the returned boundary's clean re-evaluation counts.

## Measured (2026-07-23, this box, single-threaded)
- very_low_fidelity + Boozer/QI, no turbulent (= train): ~1.4 s/eval on mp=1 boundaries, 2-8 s once modes are inflated (mp=2-3). One-time ~9 s jax JIT per process, ~4 s import.
- low_fidelity + QI (= val gate): ~2.2 s.
- official high fidelity (= private truth, SimpleToBuildQIStellarator.evaluate): ~64 s for an mp=1 boundary, up to ~128 s+ for high-mode ones.
- vlf vs lf agreed to 4 decimals on feasibility for the NAE seed; official high-fid disagreed by ~0.03 (1.082 -> 1.110). Expect the gap to GROW with boundary complexity.

## Why it matters
- Train budget: default 72 evals AND 240 s CPU; the s17 seed-bank campaign runs at 160 evals / 480 s (STELLAR_TRAIN_OVERRIDES). Bank-era caveat: high-mode bank boundaries cost 12-27 s/eval at vlf, so the effective eval count is far lower — see seed-bank-regime.md. fm.eval_many on 2 workers ~1.36x faster wall-clock than sequential (stragglers dominate; batches of 2).
- Low->high-fidelity disagreement is a documented failure mode of this benchmark (paper §resolution gaming): a candidate that looks feasible at vlf can be infeasible at high fid. Large vlf/lf vs private disagreement on an elite = red flag, log it, never trust the cheap number.

## Verdict
promising — exploit the dial (search cheap, verify dear), but treat every cheap-fidelity feasibility claim as provisional.
