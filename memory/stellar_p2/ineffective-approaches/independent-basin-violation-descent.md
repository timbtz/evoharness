# Independent Basin Descent from NAE/Ellipse Seeds
In an isolated state without the public seed bank, dynamically generated NAE/ellipse seeds lack the baseline `objective_L` to be competitive and are capped at negative scores. All structural contractions, coordinate-axis, SPSA, forward-probe, and physics-first constraint-targeted transforms fail to close the massive physics gap, often *increasing* the constraint violation.

## How it was tried
- `stellar_p2-s207-18745292` c0001 (ACC, train -0.5619): The baseline. Generated a 15-candidate portfolio of NAE/ellipse seeds at varying aspect ratios and sorted strictly by max constraint violation. Established the floor.
- `stellar_p2-s207-18745292` c0011r1 (REJ, train -0.975): 12-seed NAE portfolio triaged by violation, followed by an R/Z-split structural contraction sweep. Regressed because structural contraction *increases* aspect ratio violation on already-infeasible seeds.
- `stellar_p2-s207-18745292` c0012 (REJ, train -0.983): Coordinate-axis descent targeting the dominant constraint. Single-axis moves failed to locate the feasibility gradient and regressed further into negative scores.
- `stellar_p2-s207-18745292` c0013 (REJ, train -0.836): Batched constraint-targeted contraction sweep guided by per-individual worst-violating-constraint diagnostics. Failed to achieve violation < 0.3.
- `stellar_p2-s207-18745292` c0014 (ACC, train -0.989): Replaced iterative SPSA with a fully batched R/Z-split structural contraction sweep. Regressed heavily relative to the baseline floor.
- `stellar_p2-s207-18745292` c0015 (ACC, train -0.563): Prepended a single batched physics-first sweep to the parent SPSA loop. Safely matched the parent floor.
- `stellar_p2-s207-18745292` c0016 (REJ, train -0.836): Added a second parallel basin to the SPSA loop. Splitting the budget between two basins failed to improve violation.
- `stellar_p2-s207-18745292` c0017 (REJ, train -0.624): Widened the initial portfolio diversity (nfp 2-5, AR 6-9, mpol 1-3) but kept the SPSA loop intact. Regressed slightly.
- `stellar_p2-s207-18745292` c0018 (REJ, train -0.590): Replaced SPSA antithetic pairs with a single-eval forward probing scheme. Regressed relative to the parent floor.
- `stellar_p2-s207-18745292` c0019 (REJ, train -0.649): Replaced the SPSA loop with a few batched grid sweeps of targeted single-mode perturbations (poloidal depth, toroidal shift). Regressed relative to the parent floor.

## Why it failed
Writers repeatedly predicted that SPSA loops, single-eval forward probing, or structural contraction sweeps would systematically minimize constraint violations and cross the 0.01 feasibility threshold. The code applied these exact mechanisms. However, dynamically generated NAE/ellipse seeds start at enormous baseline violations (>0.5). Structural contractions (which are designed to push the aspect ratio to the tolerance wall) applied to boundaries that are already massively violating the aspect ratio constraint strictly *increase* the violation. Furthermore, random-direction or forward-probe SPSA fails to locate the feasibility gradient because the infeasible landscape is flat/degenerate relative to the 72-eval budget.

## Verdict
exhausted — Stop attempting to build independent basins from NAE/ellipse seeds or applying structural contraction to them. The 0.01 feasibility wall is uncrossable for random low-mode seeds within the 72-eval budget.
