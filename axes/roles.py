"""S5 Roles: single_strong | split_roles. Model ids come from config, never from code.

The field's open question (no paper tests model-to-role assignment) — the ledger attributes
every call to a role so the screening study can measure this switch.
"""
from __future__ import annotations


class SingleStrong:
    def __init__(self, models: dict):
        self.models = models

    def writer(self) -> str:
        return self.models["strong"]

    def feedback_model(self) -> str:
        return self.models["strong"]


class SplitRoles(SingleStrong):
    """Cheap model writes code (high volume); strong model reflects/judges (low volume).
    AlphaEvolve's Flash(breadth)+Pro(depth) pattern."""

    def writer(self) -> str:
        return self.models["cheap"]
