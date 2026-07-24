Stellarator boundary optimization (ConStellaration P2, "simple-to-build QI
stellarator"). You are NOT evolving a solution — you are evolving an OPTIMIZER:
a Python module (stdlib + numpy + scipy + nevergrad; no files/network) exporting

    def solve(fm, rng) -> dict   # a boundary: {"r_cos": [[...]], "z_sin": [[...]],
                                 #  "n_field_periods": int}  (stellarator-symmetric)

It searches the ~10-80-D Fourier space of plasma boundaries under a HARD budget
of metered physics-simulator calls and returns the best boundary it found.

The fm handle (your only access to physics; every eval costs ~1.5 s CPU):
- fm.eval(boundary_dict) -> dict of metrics, or None on failure with the reason
  in fm.last_error (VMEC non-convergence etc. — informative, not fatal).
  Budget: 72 evals AND a 240 s CPU deadline, whichever hits first. A single
  eval is hard-killed after 60 s (pathological boundaries can hang the
  solver) — it still costs a budget unit and returns None.
- fm.eval_many([b1, b2, ...]) -> list of metrics/None, same order. Runs on 2
  parallel workers: a batch of 2+ costs ~half the wall-clock of sequential
  fm.eval calls but the SAME budget units — batch/population strategies fit
  far more search into the CPU deadline.
- Both accept fidelity="low_fidelity" (~2x slower, tighter force tolerance =
  what the survival gate uses). The default train fidelity is BLIND to small
  perturbations (a train score identical to the parent's after a change means
  UNTESTED, not safe) — self-verify your final boundary at low_fidelity
  before returning it; a landmine boundary scores zero at the gate.
- fm.score(metrics) -> float, the objective you maximize ("shaped score"):
  official P2 score in (0..1] if all constraints hold, else -(max normalized
  constraint violation) (negative; -0.02 is nearly feasible, -1.0 is far off).
- fm.remaining() -> evals left.
- fm.seed_nae(aspect_ratio=, max_elongation=, rotational_transform=,
  mirror_ratio=, n_field_periods=, max_poloidal_mode=, max_toroidal_mode=)
  -> ~QI near-axis boundary dict (free, unmetered). The right seed family.
- fm.seed_ellipse(aspect_ratio=, elongation=, rotational_transform=,
  n_field_periods=) -> rotating-ellipse boundary dict (free).
- fm.seed_bank(i) -> boundary dict from a bank of PUBLIC leaderboard
  submissions (free); fm.seed_bank_info() -> their official high-fidelity
  scores/feasibilities. Best bank entry: official 0.636. WARNING: high-mode
  bank boundaries cost 12-27 s PER EVAL at train fidelity (vs 1.4 s for
  simple ones) — budget accordingly. Returning a bank boundary unimproved
  (or trivially perturbed) is WORTHLESS: the novelty bar is beating the
  bank's best. Recombining several bank seeds (coefficient blending,
  mode grafting between them) and polishing L while staying feasible is
  the open game; feasibility margins are thin (some bank seeds sit at
  0.007-0.010 of the 0.010 tolerance — one is already infeasible at train
  fidelity).

Objective: maximize L = minimum normalized magnetic-gradient scale length
(metrics key: minimum_normalized_magnetic_gradient_scale_length; score = L/20
clipped to [0,1]). Hard constraints (feasible iff every normalized violation
<= 0.01): aspect_ratio <= 10; |edge_rotational_transform_over_n_field_periods|
>= 0.25; log10(qi) <= -4; edge_magnetic_mirror_ratio <= 0.2;
max_elongation <= 5. metrics contains all of these plus feasibility, p2_score,
shaped_score. The QI residual is typically the hardest constraint by orders of
magnitude; seeds start infeasible (shaped ~= -0.5) and driving feasibility down
IS the game until the wall is crossed.

Boundary encoding: r_cos[m][n+ntor], z_sin[m][n+ntor] give R,Z Fourier
coefficients of angle (m*theta - nfp*n*phi), rows m=0..mpol, cols n=-ntor..ntor.
Keep r_cos[0][ntor] (major radius, ~1.0) fixed; row 0 entries left of center
(and z_sin[0][<=ntor]) must stay 0 by symmetry — copy the pattern of a seed.
Matrices may be zero-padded to more rows/cols to unlock finer modes (mode
continuation: coarse first, inflate later, proven trick).

Scoring of YOUR program: after solve() returns, the harness re-evaluates the
RETURNED boundary itself (uncounted) — that shaped score is your train fitness.
Accepted candidates are re-verified on a clean higher-fidelity run of the
returned boundary only (val); final truth is the official high-fidelity
evaluator (~128 s, private). Forged fm bookkeeping cannot help you; only the
boundary you return matters. Calling constellaration.forward_model directly
instead of fm.eval is a disqualifying cheat (audited) — all physics through fm.
Same rng seed + same code => identical result (simulator is deterministic);
timing out (240 s CPU) or crashing scores -inf, so return your best-so-far
early rather than gambling the deadline. Every good boundary you evaluate is
also harvested into a persistent cross-run archive — finding several distinct
good regions beats polishing one mediocre one.
