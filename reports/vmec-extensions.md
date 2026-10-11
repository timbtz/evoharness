# How I extended the VMEC++ feedback and optimization loop

[← Project overview](../README.md#the-vmec-case-study) · [Gradient evidence](data/vmec-gradient-audit.json) · [Feedback evidence](data/vmec-feedback-audit.json)

I added **richer evaluator feedback, boundary-gradient tools, and constrained local optimization** around ConStellaration and VMEC++. The work progressed from explaining failed proposals to estimating useful directions, then repairing several competing constraints together. The pinned reference remains **ConStellaration 0.2.6 / VMEC++ 0.4.11**; these extensions do not implement a new equilibrium solver or a VMEC adjoint.

![VMEC++ at the center of four extensions: cheap boundary tools, richer feedback, hybrid gradients, and constrained updates. Fresh evaluation decides whether to accept a proposed step.](figures/vmec-extensions-v3.png)

<sub>Source and saved-run audit: 2026-10-11, revision `a855ae5`. This note separates historical measurements from current implementation and source-level concerns. No new physics evaluations were run for this audit.</sub>

## 1. Make the feedback useful before a boundary is feasible

The official P2 score is `L / 20` only if every normalized constraint violation is at most **0.01**; otherwise it is zero. Here, `L` is the minimum normalized magnetic-gradient scale length. A zero score alone does not tell the optimizer what to repair.

| Addition | What the optimizer receives | Why I added it |
|---|---|---|
| **Constraint feedback** | Objective, all five violations, limiting constraint; negative worst violation for infeasible search candidates | Distinguish progress before official feasibility |
| **Graded nonconvergence** | Available force residuals, tolerance, iterations, and a bounded failure score | Retain information that a timeout or `None` would discard |
| **Margin feedback** | For scored feasible candidates: `L/20 − 0.92 × max(0, worst_violation − target)`; default target `0.002` | Make spending feasibility tolerance visible during selection |
| **Provenance feedback** | Scale-normalized distance and similarity to the seed bank | Prevent uniform rescaling from masquerading as novelty |
| **Feedback-first context** | Family outcomes, latest 24 measurements, failure classes, active constraints | Put measured evidence ahead of long plans |

These search signals leave the official score unchanged. Infeasible candidate selection remains based on negative worst violation. Standalone polish also logs an ungated `L/20` margin surrogate; its feasibility mode accepts by violation reduction. Soft failures occupy a separate low range, **[−1001, −1000]**, and do not become converged archive entries. Novelty penalties are additional search choices; distance from a seed alone does not establish scientific novelty.

The context change fixed a concrete information-flow failure: a roughly **26,000-character plan** had preceded evidence in a prompt truncated at **12,000 characters**. The measurements never reached the writer. This explains the change, but is not a controlled demonstration of improved scores.

Sources: [task and evaluator interface](../src/evoharness/tasks/stellar_p2/task.py) · [context assembly](../src/evoharness/tasks/stellar_p2/discovery/evidence.py).

## 2. Add directions the optimizer can actually follow

I first exposed `fm.aspect`, `fm.margin_grad`, and `fm.margin_step`: boundary-only geometry and a small Newton correction, without an equilibrium solve. The later `fm.metric_grad` service estimates how equilibrium-dependent metrics change with boundary coefficients. `fm.grad_step` converts those estimates into a proposed boundary; it does not certify improvement.

### The hybrid derivative

For selected Fourier coefficients, I run the pinned solver at positive and negative perturbations. JAX differentiates the downstream metrics with respect to the resulting equilibrium or Boozer arrays. The two factors are contracted by the chain rule:

```text
boundary direction ≈ metric derivative with respect to solver outputs
                     × central differences of solver outputs
```

This gives approximate boundary gradients without differentiating through the solver itself.

| Metric / constraint | Implemented derivative | Extra equilibrium solves beyond the shared stencil |
|---|---|---:|
| **Aspect** | Exact derivative of boundary geometry | 0 |
| **Objective L** | JAX metric derivative × differences of six equilibrium arrays | Shared stencil |
| **log₁₀(QI)** | JAX derivative × differences of Boozer arrays and transform values | 0 |
| **Mirror ratio** | JAX derivative × the same magnetic-field differences | 0 |
| **Elongation** | Boundary derivative with the magnetic axis held fixed | 0; approximate |
| **Rotational-transform constraint** | Measured and checked; no dedicated constraint gradient | — |

Mirror was added when QI-only repair worsened mirror violation. Elongation support addressed another obstacle in imported configurations. Mirror is available in the service, but the candidate API's default request still omits it: it must be requested explicitly.

### Cost and implementation boundaries

- **41 reserved evaluation units** for `k=20`: `2k+1`, charged before the service runs. With only `m≤k` eligible coefficients, a completed stencil uses `2m+1` solves. Trial verification costs another solve.
- The stencil selects large, **already-nonzero, symmetry-permitted coefficients**. New modes can change grid dimensions; incompatible columns are dropped. This is a local refinement mechanism, not a mechanism for discovering new spectral support.
- JAX runs in a **separate child process**. Initializing it in the parent conflicted with the forked evaluator pool after timeouts; boundary-only tools keep the parent NumPy-only.

Sources: [gradient service](../src/evoharness/tasks/stellar_p2/physics/metrics/boundary_grad.py) · [finite differences](../src/evoharness/tasks/stellar_p2/physics/finite_difference.py) · [metric derivatives and polish](../src/evoharness/tasks/stellar_p2/physics/polish.py).

## 3. Move from local ascent to joint constraint repair

There are two different aims: **increase the margin-adjusted objective** near a feasible design, or **reduce infeasibility** in an imported design. Calling both “gradient descent on the official score” would obscure that distinction.

| Design change | Reasoning from the runs | Current behavior |
|---|---|---|
| **Small, verified steps** | Larger steps changed the active minimum and made predictions unreliable | Cap coefficient movement, evaluate again, shrink after rejection |
| **Objective alignment** | Projecting away every constraint direction could disagree with the selection score | Include the margin-penalty derivative when aspect or QI is limiting; use supported constraint projections otherwise |
| **Joint projection** | Sequential projections could reintroduce a previously removed component | Solve a small Gram system for simultaneous constraint handling |
| **Joint feasibility descent** | QI-only and mirror-only repair alternated the bottleneck; averaging opposing gradients stalled | Solve for a minimum-norm direction that reduces near-tied supported violations together |
| **Incumbent-based acceptance** | Comparing against the previous rejected record could accept a regression | Compare with the accepted incumbent and preserve the best boundary |

In repair mode, rows of `A` are normalized constraint gradients. The regularized joint solve approximates **`A × step = −1`**, followed by a cap and fresh evaluation. Constraints within `0.15` of the largest supported violation participate; aspect can also be held near its bound. A repair step must reduce the measured worst violation by more than `1e−6`.

The `qidesc-platform` trajectory records the progression from individual QI/mirror steps to joint descent. The two gradients became strongly opposed, with cosine approximately **−0.805**. This explains why simply fixing the current worst constraint was insufficient.

The early successful polish predates some of these corrections. Its step 8 accepted a regression from **0.6225263 to 0.6223879**; the current acceptance logic addresses that bug. Accepted-event counts therefore must not be presented as a monotonic improvement curve. The candidate API can also add a novelty direction after projection, so its step is still only a proposal—not a feasibility guarantee.

## 4. What the saved runs actually show

| Experiment | Measured change | Attempts / accepted | Solver calls | Interpretation |
|---|---|---:|---:|---|
| **Local `qi_polish`** | Official P2 **0.632082 → 0.634038** | 17 / 11 | **715** | Verified local gain; official violation also fell **0.004728 → 0.004361** |
| **Relaxed-margin ceiling** | Final official P2 **0.634421**, violation **0.001924** | 29 / 23 | 1,219 | Relaxing the target to `0.009` still ended in trust collapse |
| **QI-platform repair** | Worst violation **0.558492 → 0.465618** | 43 / 31 | 4,215 | Partial repair; objective L fell **13.674 → 13.369** |
| **Goodman repair** | Worst violation **0.208670 → 0.150911** | 21 / 13 | 1,259 | Partial repair; L stayed near **6.09** |
| **Second Goodman repair** | Worst violation **0.146401 → 0.146401** | 6 / 0 | 397 | No accepted improvement |

For the first run, the cost reconciles exactly: **one initial solve + 17 × (41 gradient solves + one trial solve) = 715**. Its official gain of **0.001956** is evidence of refinement within an inherited design's neighborhood.

All three repair endpoints remain above **0.01**. They do not establish a newly feasible design. The later **0.04070** frontier in the README belongs to a separate LLM repair campaign and must not be credited to these gradient runs.

### Did giving the LLM gradients improve the winner?

The three full-gradient ledgers contain **136 candidate source records**. A fresh static AST audit finds gradient-tool calls in **15/51**, **12/57**, and **3/28** records: **30/136** overall. Eight records did not parse. These counts show calls present in source, not confirmed execution. Four accepted gradient-containing candidates only tied their incumbents; all three final `best.py` artifacts lack these calls.

The earlier aspect-only A/B also had mixed outcomes. Full-gradient campaigns increased candidate budgets from **160 evaluations / 480 seconds** to **400 / 2,400**, so their endpoints are not a matched comparison with that earlier experiment. Tool availability and a successful standalone polish do not yet establish a benefit from LLM-directed gradient use.

## 5. What the deeper audit changed in my interpretation

| Finding | What can safely be concluded |
|---|---|
| **Step-size evidence is specific** | The prediction ratio **0.99 at `3e−5`**, versus **−0.05 at `1e−4`**, concerns a QI-descent probe. It is not universal accuracy for every metric or boundary. |
| **Cause of stalling remains unresolved** | Early notes blamed a QI wall; later review blamed the margin penalty. The relaxed-margin run also stalled, so neither is a complete demonstrated explanation. |
| **Cross-version derivatives need a fair test** | An early sidecar test changed both solver and downstream functional. The archived follow-up reports Jacobian disagreement of **13–34%**, below pinned finite-difference truncation of **27–44%**. Foreign derivatives may propose directions; pinned evaluation decides success. |
| **Faster solves can change the result** | Single-stage training evaluation was identical in **24/31** paired cases, with maximum L drift **0.041809** and median speedup **1.34×**. True restart was identical in **10/31**, with maximum drift **3.755960**. Fresh strict verification remains necessary. |

Two additional **source-level concerns need numerical validation**: the L derivative holds `Aminor_p` fixed although it appears in normalization, and repair mode divides the aspect gradient by 10 after the helper has already normalized it. Neither has been established as the cause of a recorded stall. The frozen-axis elongation approximation and missing rotational-transform constraint gradient further limit what the tools can promise.

## 6. Next experiments

I would first validate the derivative normalization and active-constraint handling on fixed boundaries, then compare gradient-free, geometry-only, and full-gradient search under **equal total solver budgets**, across seeds. More compute should fund these controls and official verification as well as more proposals. The unresolved research question is whether better local tools help an LLM find useful new designs, beyond improving an inherited one.

The linked evidence exports preserve selected measurements, raw local source paths, hashes, and audit methods. Full local run directories are not included in the public repository; the exports do not substitute for a complete replay archive. [Broader research history and setup](research-notes.md) · [Illustration prompt](data/visual-provenance.json).
