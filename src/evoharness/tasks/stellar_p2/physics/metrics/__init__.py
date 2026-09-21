"""Differentiable ConStellaration P2 score machinery (Plan 2, 2026-08-01).

Everything here is proposal/analysis machinery: the pinned in-image evaluator
remains the only accept/rank authority. Modules:

- margins_jax:  JAX values + gradients for the five P2 constraint margins
                (aspect exactly boundary-only; elongation boundary+axis;
                edge_iota/mirror/L-grad-B from wout arrays).
- qi_jax:       JAX port of the scored QI residual (Goodman 2023), Boozer-array
                inputs, `smoothing` parameter; smoothing->0 matches the pinned
                qi.py.
- feasibility:  smooth feasibility aggregates for internal fitness only, with
                a strict-dominance guard vs officially-feasible results.
- difftest:     corpus loader, pinned-oracle cache builder, parity/gradient
                reporting. Runs inside the evoharness-stellar-eval image.
"""
