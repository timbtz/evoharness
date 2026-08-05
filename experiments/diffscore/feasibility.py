"""Deliverable C: smooth feasibility aggregates for INTERNAL fitness only.

The pinned evaluator's hard gate (max violation <= 1e-2) stays the only
accept/rank authority. These smooth surrogates exist so a local polish loop
can follow gradients through the gate region instead of seeing a step. The
strict-dominance guard guarantees nothing smooth ever outranks an
officially-feasible-and-scored result in any fitness comparison.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

GATE = 1e-2  # problems.py:11 _DEFAULT_RELATIVE_TOLERANCE


def soft_feasibility(violations, tau=1e-3):
    """Smooth upper bound of compute_feasibility's max(violations, 0):
    tau * logsumexp(v/tau) over the violations and 0. -> max as tau -> 0."""
    v = jnp.concatenate([jnp.atleast_1d(violations), jnp.zeros(1)])
    return tau * jax.nn.logsumexp(v / tau)


def smooth_fitness(score_estimate, violations, tau=1e-3, slope=1.0,
                   margin_target=2e-3):
    """Internal, differentiable stand-in for the shaped score: the score
    estimate minus a penalty on feasibility spent beyond margin_target
    (mirrors task.py _honest) plus a softplus barrier on the official gate."""
    f = soft_feasibility(violations, tau)
    penalty = slope * jax.nn.softplus((f - margin_target) / tau) * tau
    barrier = jax.nn.softplus((f - GATE) / tau) * tau
    return score_estimate - penalty - 100.0 * barrier


def rank_key(officially_feasible: bool, official_score: float,
             smooth_score: float) -> tuple:
    """Total-order key for internal candidate ranking. Any officially
    feasible + scored result strictly dominates any smooth-only estimate,
    regardless of magnitudes."""
    return (1 if officially_feasible else 0,
            official_score if officially_feasible else smooth_score)
