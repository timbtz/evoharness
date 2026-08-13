Stellarator boundary optimization (ConStellaration P2, "simple-to-build QI
stellarator"). You are NOT evolving a solution — you are evolving an OPTIMIZER:
a Python module (stdlib + numpy + scipy + nevergrad; no files/network) exporting

    def solve(fm, rng) -> dict   # a boundary: {"r_cos": [[...]], "z_sin": [[...]],
                                 #  "n_field_periods": int}  (stellarator-symmetric)

It searches the ~10-80-D Fourier space of plasma boundaries under a HARD budget
of metered physics-simulator calls and returns the best boundary it found.

The fm handle (your only access to physics; every eval costs ~1.5 s CPU):
- fm.eval(boundary_dict) -> dict of metrics, or None on failure with the reason
  in fm.last_error (solver crash etc. — informative, not fatal).
  Budget: 72 evals AND a 240 s CPU deadline, whichever hits first. A single
  eval is hard-killed after 60 s (pathological boundaries can hang the
  solver) — it still costs a budget unit and returns None.
  A solve that runs but does NOT converge returns a dict with
  "soft_fail": True and a graded sentinel score in [-1001, -1000] (higher =
  closer to convergence; residuals in "fsqr"/"fsqz"/"fsql"). It is strictly
  worse than every converged result — never keep such a boundary, but the
  gradient across soft-fails tells you which direction is closer to solvable.
- Evals of boundaries within 1e-3 (max coefficient) of a recently converged
  one skip the solver's coarse init stage and run ~25-45% faster (identical
  metrics) — small-step local search literally buys more evals per second
  than blind jumps.
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
- fm.seed_nae_axis(...same physical parameters..., torsion=,
  max_poloidal_mode=3, max_toroidal_mode=3) -> independent near-axis boundary
  with magnetic-axis torsion exposed rather than fixed to 1.33/aspect. Use this
  for low-dimensional axis/boundary co-design; construction is free, every
  resulting boundary still needs a fresh metered evaluation.
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

NOVELTY REQUIREMENT (2026-07-24): a boundary within max-coefficient distance
1e-3 of ANY same-nfp public seed-bank entry is a NEAR-COPY of another user's
submission — the export guard refuses it, and the harness subtracts up to 0.05
from feasible train/val scores inside that ball (metrics report bank_dist and
novelty_penalty; the official private score is never shaped). The goal is a
boundary that beats 0.6361 officially AND sits >= 1e-3 from every bank seed:
micro-polishing a bank seed CANNOT produce a submittable result. Promising
paths: recombination / mode-grafting between different bank optima, coordinated
multi-mode moves that leave the ball while holding feasibility, or an
independent basin grown from NAE seeds. Measure your own distance in-run by
comparing padded coefficient matrices against each fm.seed_bank(i).

fm.bank_dist(boundary) (free, no eval budget) returns the exact max-coefficient
distance to the nearest same-nfp bank seed — the same metric as the harness
penalty and the export guard. fm.score() is BLIND to the novelty penalty: your
acceptance key must combine them yourself, e.g.
key = score - 0.05 * max(0.0, 1.0 - fm.bank_dist(b) / 1e-3).

FEASIBILITY MARGIN (2026-07-27, changes what "better" means): a design is
feasible while its worst normalized constraint violation stays <= 0.01, and the
score rises ~0.92 per unit of that budget you spend — so pushing aspect ratio
into the wall raises the raw score without improving the physics. An audit of the
previous campaign found the entire margin over the leaderboard bar came from
this, not from better structure. Train/val fitness is now the score DISCOUNTED to
a 0.002 margin:
    honest = p2_score - 0.92 * max(0.0, feasibility - 0.002)
Every metrics dict returned by fm.eval/eval_many carries "honest_score" already
computed — select on it, not on p2_score. The private/official score stays the
raw number, so a run's headline result is undistorted. Practical consequence:
squeezing the last 0.001 of aspect-ratio tolerance is now worth roughly nothing,
and a boundary that reaches the same L at feasibility 0.002 beats one that needs
0.009. Raise L by structure.

<!--MARGIN-GRAD-->
EXACT ASPECT GRADIENT (2026-08-05, new — free, no eval budget, no solver call):
aspect_ratio is the one scored constraint that depends on the boundary ALONE.
It is now computed exactly (~1e-13 against the simulator) and differentiated in
closed form, so the margin that is almost always binding costs zero evals to
read and to place:
- fm.aspect(b) -> float. The exact aspect ratio the simulator would report.
  Screen candidate boundaries before spending an eval on them.
- fm.margin_grad(b) -> {"aspect", "violation", "grad_r_cos", "grad_z_sin"}:
  violation = (aspect - 10)/10 (the normalized number the feasibility rule
  uses), and the gradients are d(violation)/d(coefficient) as matrices shaped
  like r_cos/z_sin, already zeroed where symmetry pins a coefficient.
- fm.margin_step(b, target=0.002, cap=3e-3) -> a new boundary whose aspect
  violation IS target (an exact Newton walk, not a linear guess), moved by at
  most `cap` in max-coefficient distance.
Two things this is for. (1) Margin placement: honest fitness discounts every
unit of feasibility above 0.002, so put the geometric margin exactly at 0.002
in one free call instead of bisecting it with a ladder of evals. (2) Constant-
margin structure search: project any structural move you invent onto the
subspace orthogonal to grad_r_cos/grad_z_sin (x -= g * (x.g)/(g.g), flattening
both matrices into one vector) and it changes shape WITHOUT moving the aspect
margin — that is how you raise L without paying tolerance.
MEASURED CAVEAT, take it seriously: the qi margin is coupled and moves AGAINST
aspect at 1.5-3x strength in violation units on tight boundaries, and qi is NOT
modelled here (it needs an equilibrium solve). A stepped boundary can land its
aspect margin perfectly and still be infeasible on qi. Nothing is verified
until fm.eval says so; use the free tools to spend evals on the constraints
that actually need them. Each call costs ~0.1-0.5 s of your CPU deadline on
high-mode boundaries — batch your bookkeeping, don't call them per candidate in
an inner loop of thousands.
<!--/MARGIN-GRAD-->

<!--SOLVE-GRAD-->
SOLVER GRADIENTS — THE OBJECTIVE AND qi, w.r.t. THE BOUNDARY (2026-08-10, new):
the aspect tools above give you the constraint that is easy and, on a polished
boundary, usually NOT the one blocking you. These give you the two that are:
the objective L-gradB itself, and qi.
- fm.metric_grad(b, k=20, h=3e-5) -> dict, or None (reason in fm.last_error).
  COSTS 2*k+1 EVALS (41 by default, ~12 minutes). It is not free and there is
  no cheaper way: no adjoint exists, so the boundary derivative is finite
  differences of the solver's output arrays, contracted against exact analytic
  gradients of the metrics. Keys:
    grad_L          d(L-gradB)/d(coeff)        <- the objective. Ascend this.
    grad_qi         d(log10 qi)/d(coeff)
    grad_aspect     d(aspect violation)/d(coeff)   (exact, came for free)
    grad_elongation d(elongation)/d(coeff)         (axis frozen: a direction,
                    not a prediction)
    base            honest_score / L / feasibility / violations / qi / aspect
    solves, trust, coeffs
  Each gradient is {"r_cos": [[...]], "z_sin": [[...]]}, shaped like the
  boundary, zero where symmetry pins a coefficient.
- fm.grad_step(b, grad, cap=3e-5, novelty_weight=None) -> a new boundary that
  ascends the SAME composite used for selection: L/20 minus the 0.92 penalty on
  the currently worst violation above the margin target. For aspect and qi the
  exact available gradient is used (including qi's /4 violation normalization);
  unsupported active constraints fall back conservatively. Inside the novelty
  ramp it also follows a scale-normalized bank-distance subgradient by default;
  pass an explicit nonnegative novelty_weight to tune or disable that term.
  Free. A proposal, not a result — fm.eval decides.

THE STEP SIZE IS THE WHOLE GAME. Read this before choosing `cap`:
the trust region is 3e-5 in max-coefficient distance, and it is MEASURED, not a
safety margin. Predicted-vs-actual improvement is 0.99 at cap 3e-5 and -0.05 at
1e-4 — one order larger and the step goes the WRONG WAY. The reason is the
landscape, not the estimator: the objective is a min over a grid whose argmin
switches under 1e-4 boundary steps. A previous campaign's writers used the
aspect tool at caps of 2e-3 to 4e-3, which is 100x outside this region, and
those runs never beat the boundary they started from. Do not reuse that habit
here. If you want to move further, take MANY small steps and re-linearize —
that is what a gradient is for — or take a big structural jump and stop
pretending the gradient describes it.
A working loop, if you want one: evaluate, metric_grad, grad_step at 3e-5,
evaluate, keep it if honest_score rose, else halve the cap and retry. Six such
steps took a champion boundary from honest 0.61996 to 0.62253 with every step
accepted and the feasibility margin held to within 1e-4.
BUDGET IT DELIBERATELY: at 41 evals a gradient, three re-linearizations are
most of a 160-eval budget. Either raise your own eval efficiency elsewhere, or
spend the gradient where it decides something — on the incumbent you actually
intend to ship, not on every candidate you dream up. k selects how many
coefficients are differentiated (the k largest free nonzero ones); a smaller k
is a cheaper, blinder gradient, and the ones it drops are treated as zero.
<!--/SOLVE-GRAD-->

SHAPE NOVELTY, NOT JUST DISTANCE (2026-07-27): max-coefficient distance is a
weak novelty test. The previous campaign's champion cleared the 1e-3 export ball
at bank_dist 2.6e-3 while being cosine 0.999989 to the public #1 boundary — the
same shape, rescaled. Every val/private metrics dict now also reports "bank_cos",
the cosine to the nearest same-nfp bank seed, and this campaign widens the
train/val novelty ramp beyond the export bar (see STELLAR_NOVELTY), so the
penalty keeps pointing out of the public basin instead of switching off the
moment a candidate is barely exportable. A genuinely different design shows up
as a bank_cos meaningfully below 0.9999, not merely a distance above 1e-3.
