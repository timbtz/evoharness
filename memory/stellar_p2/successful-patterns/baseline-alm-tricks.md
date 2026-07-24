# What the official ALM baseline does (P2 = 0.431)
The published augmented-Lagrangian + Nevergrad baseline reached 0.431 in 34 h x 96 vCPU; its tricks transfer to any optimizer evolved here.

## How it was tried (source: proximafusion/constellaration optimization_examples, MIT)
- Per-mode exponential preconditioning: parameter scale ~ 1.5^-(|m|+|n|) ("infinity_norm_spectrum_scaling: 1.5") — high modes move less. The seed optimizer copies exactly this damping.
- Mode space 4x4 (max_poloidal_mode = max_toroidal_mode = 4), NAE initial guess (nfp=3, aspect_ratio=10, max_elongation=5, rt=0.75, mirror=0.25 — note: deliberately infeasible start, even above the mirror bound).
- ALM outer loop: penalty parameters x5 on violation, trust-region bounds x0.95 shrink (0.5 -> 0.05 floor), tolerance x0.8 per round; inner oracle = Nevergrad NGOpt, budget 1500 -> 20000 per round (+260/round).
- QI constraint handled in log10 space (log10(qi) <= -4), NOT raw residual — raw spans orders of magnitude and swamps the penalty balance.
- low_fidelity forward model for the whole search; official evaluate only at the end.

## Why it worked
Feasibility-first shaping: ALM drives max violation down before polishing L; the spectrum scaling keeps early moves in the smooth low-mode subspace.

## Verdict
promising — the reference recipe. Evolving BETTER outer-loop logic (penalty schedules, fidelity switching, restarts) than ALM's fixed schedule is exactly the headroom this task exists to explore (ExLLM got 0.505 with far fewer evals; leaderboard #1 is 0.636).
