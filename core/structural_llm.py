"""Structured z.ai proposal/critique control plane for structural discovery."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from core.structural_discovery import OperatorSpec, stable_hash

FAMILIES = {"nae", "nae_axis", "ellipse", "mode_continuation", "truncation_reconstruction",
            "structural_dilation", "axis_boundary_codesign", "nfp_pivot"}
CONSTRUCTION_FAMILIES = {"nae", "nae_axis", "ellipse"}

WRITER_PROMPT = """You design ONE structural stellarator-basin operator, not a whole optimizer.
Target a 5-10% or larger increase in L. Gradients and coefficient micro-polish are forbidden.
Public boundaries and inherited artifacts are forbidden: this is an independent arm.
Allowed families: {families}.

Return ONLY one JSON object with exactly these fields:
family, version, parameters, mechanism, expected_l_gain_fraction, expected_constraint_effects,
novelty_mechanism, kill_criterion, falsifiable_prediction.
`parameters` must be numeric/bool/string JSON values. For nae use the documented construction
parameters EXACTLY: aspect_ratio, max_elongation, rotational_transform, mirror_ratio,
n_field_periods (2 or 3), max_poloidal_mode=3, max_toroidal_mode=3. The two mode
values MUST be the integer 3, never 4 or higher. For ellipse use EXACTLY:
aspect_ratio, elongation, rotational_transform, n_field_periods (2 or 3).
For nae_axis use the NAE keys plus torsion in [0.03, 0.35]. This is the preferred
axis/boundary co-design family; vary torsion independently of aspect ratio.
`expected_constraint_effects` must have EXACTLY these five keys:
aspect, iota, qi, mirror, elongation. Do not rename them or discuss other constraints there.
Do not include source code, boundary coefficients, seed-bank indices, file paths, or prose outside JSON.

Evidence from completed attempts (do not repeat them):
{evidence}
"""

CRITIC_PROMPT = """Act as a hostile scientific and security reviewer. Review the proposed typed
operator below. Reject it if it is local polish, predicts <5% L gain, has no physical mechanism,
uses public/inherited artifacts, duplicates the evidence, lacks a cheap kill criterion, or its
parameters cannot be executed. Return ONLY JSON:
{{"accept": boolean, "reasons": [strings], "risk": "low|medium|high"}}

Proposal:
{proposal}

Prior evidence:
{evidence}
"""

REPAIR_PROMPT = """Your JSON proposal was rejected before physics for this exact error:
{error}
Return a corrected proposal ONLY as JSON. Preserve the physical hypothesis but obey the schema
literally. Allowed family for this round: {families}. NAE/NAE_AXIS modes MUST both equal 3.
Expected constraint keys MUST be exactly aspect,iota,qi,mirror,elongation. Do not add parameters.
Previous invalid response:
{raw}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("response contains no JSON object")
        value = json.loads(match.group())
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


@dataclass(frozen=True)
class Proposal:
    operator: OperatorSpec
    expected_constraint_effects: dict[str, str]
    novelty_mechanism: str
    falsifiable_prediction: str

    @classmethod
    def parse(cls, value: dict[str, Any]) -> "Proposal":
        required = {"family", "version", "parameters", "mechanism",
                    "expected_l_gain_fraction", "expected_constraint_effects",
                    "novelty_mechanism", "kill_criterion", "falsifiable_prediction"}
        missing, extra = required - value.keys(), value.keys() - required
        if missing or extra:
            raise ValueError(f"proposal schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        if value["family"] not in FAMILIES:
            raise ValueError(f"unsupported family: {value['family']}")
        if not isinstance(value["parameters"], dict) or not all(
                isinstance(v, (str, int, float, bool)) and not isinstance(v, (list, dict))
                for v in value["parameters"].values()):
            raise ValueError("parameters must contain JSON scalar values")
        forbidden = json.dumps(value["parameters"]).lower()
        if any(term in forbidden for term in ("seed_bank", "public boundary", "resume artifact",
                                               "boundary coefficients", "r_cos", "z_sin")):
            raise ValueError("independent-arm security violation")
        effects = value["expected_constraint_effects"]
        aliases = {"aspect_ratio": "aspect", "rotational_transform": "iota",
                   "qi_violation": "qi", "qa_error": "qi", "mirror_ratio": "mirror"}
        if isinstance(effects, dict):
            effects = {aliases.get(k, k): v for k, v in effects.items()}
        if not isinstance(effects, dict) or set(effects) != {
                "aspect", "iota", "qi", "mirror", "elongation"}:
            raise ValueError("all five expected constraint effects are required")
        op = OperatorSpec(value["family"], str(value["version"]), value["parameters"],
                          str(value["mechanism"]), float(value["expected_l_gain_fraction"]),
                          str(value["kill_criterion"]), independent=True)
        if op.family in CONSTRUCTION_FAMILIES:
            expected = ({"aspect_ratio", "max_elongation", "rotational_transform",
                         "mirror_ratio", "n_field_periods", "max_poloidal_mode",
                         "max_toroidal_mode"} | ({"torsion"} if op.family == "nae_axis" else set())
                        if op.family in {"nae", "nae_axis"} else
                        {"aspect_ratio", "elongation", "rotational_transform",
                         "n_field_periods"})
            if set(op.parameters) != expected:
                raise ValueError(f"non-executable {op.family} parameter schema")
            if int(op.parameters["n_field_periods"]) not in (2, 3):
                raise ValueError("construction campaign supports only NFP 2 or 3")
            if op.family in {"nae", "nae_axis"} and (
                    int(op.parameters["max_poloidal_mode"]) != 3 or
                    int(op.parameters["max_toroidal_mode"]) != 3):
                raise ValueError("construction campaign requires both modes exactly 3")
            if op.family == "nae_axis" and not (.03 <= float(op.parameters["torsion"]) <= .35):
                raise ValueError("nae_axis torsion outside [0.03, 0.35]")
        if not str(value["novelty_mechanism"]).strip() or not str(
                value["falsifiable_prediction"]).strip():
            raise ValueError("novelty mechanism and falsifiable prediction are required")
        return cls(op, {str(k): str(v) for k, v in effects.items()},
                   str(value["novelty_mechanism"]), str(value["falsifiable_prediction"]))

    def to_dict(self) -> dict[str, Any]:
        return {"operator": asdict(self.operator),
                "expected_constraint_effects": self.expected_constraint_effects,
                "novelty_mechanism": self.novelty_mechanism,
                "falsifiable_prediction": self.falsifiable_prediction}


def proposal_signature(proposal: Proposal) -> str:
    return stable_hash({"family": proposal.operator.family,
                        "parameters": proposal.operator.parameters})


class OperatorDesigner:
    def __init__(self, llm, model: str, critic_model: str, ledger,
                 temperature: float = 0.7, allowed_families: set[str] | None = None) -> None:
        self.llm, self.model, self.critic_model = llm, model, critic_model
        self.ledger, self.temperature = ledger, temperature
        self.allowed_families = allowed_families or CONSTRUCTION_FAMILIES
        self.seen: set[str] = set()

    def propose(self, evidence: str) -> tuple[Proposal | None, dict[str, Any]]:
        raw = self.llm.chat(self.model, [{"role": "user", "content": WRITER_PROMPT.format(
            families=", ".join(sorted(self.allowed_families)), evidence=evidence[:12000])}],
            self.temperature, role="structural_writer", max_tokens=3000)
        proposal = None; error = None
        for repair in range(2):
            try:
                proposal = Proposal.parse(parse_json_object(raw))
                if proposal.operator.family not in self.allowed_families:
                    raise ValueError(
                        f"family is not executable in this campaign stage: {proposal.operator.family}")
                break
            except Exception as exc:
                proposal = None
                error = exc
                if repair == 0:
                    raw = self.llm.chat(self.model, [{"role": "user", "content":
                        REPAIR_PROMPT.format(error=exc,
                            families=", ".join(sorted(self.allowed_families)), raw=raw[:6000])}],
                        .1, role="structural_schema_repair", max_tokens=3000)
        if proposal is None:
            verdict = {"accept": False, "reasons": [f"schema/security: {error}"], "risk": "high"}
            self.ledger.append({"type": "structural_proposal_rejected", **verdict})
            return None, verdict
        sig = proposal_signature(proposal)
        if sig in self.seen:
            verdict = {"accept": False, "reasons": ["duplicate proposal signature"], "risk": "low"}
            self.ledger.append({"type": "structural_proposal_rejected", "signature": sig, **verdict})
            return None, verdict
        critic_raw = self.llm.chat(self.critic_model, [{"role": "user", "content":
            CRITIC_PROMPT.format(proposal=json.dumps(proposal.to_dict(), sort_keys=True),
                                 evidence=evidence[:12000])}], .2,
            role="structural_critic", max_tokens=1500)
        try:
            verdict = parse_json_object(critic_raw)
            if set(verdict) != {"accept", "reasons", "risk"} or \
                    not isinstance(verdict["accept"], bool) or \
                    not isinstance(verdict["reasons"], list) or \
                    verdict["risk"] not in {"low", "medium", "high"}:
                raise ValueError("invalid critic schema")
        except Exception as exc:
            verdict = {"accept": False, "reasons": [f"critic schema: {exc}"], "risk": "high"}
        self.seen.add(sig)
        self.ledger.append({"type": "structural_proposal", "signature": sig,
                            "proposal": proposal.to_dict(), "verdict": verdict})
        return (proposal if verdict["accept"] else None), verdict
