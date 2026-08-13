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
# Transform families act on a SOURCE basin, so they are inherited-arm only.
# Until 2026-08-13 they were nameable but not executable: every accepted
# proposal collapsed to a construction parameter tuple, which is why elaborate
# mechanisms ("rigid rotor", "torsion dome") were never actually run.
TRANSFORM_FAMILIES = {"truncation_reconstruction", "structural_dilation",
                      "mode_continuation"}
TRANSFORM_SCHEMA = {
    "truncation_reconstruction": {"core_poloidal_mode", "core_toroidal_mode",
                                  "band_amplitude", "phase_seed"},
    "structural_dilation": {"radial_scale", "vertical_scale", "mode_decay"},
    "mode_continuation": {"max_poloidal_mode", "max_toroidal_mode",
                          "band_amplitude", "phase_seed"}}
TRANSFORM_BOUNDS = {
    "truncation_reconstruction": {"core_poloidal_mode": (1, 6), "core_toroidal_mode": (1, 6),
                                  "band_amplitude": (1e-4, .05), "phase_seed": (0, 2 ** 31)},
    "structural_dilation": {"radial_scale": (.5, 2.), "vertical_scale": (.5, 2.),
                            "mode_decay": (0., 1.)},
    "mode_continuation": {"max_poloidal_mode": (1, 8), "max_toroidal_mode": (1, 8),
                          "band_amplitude": (1e-4, .05), "phase_seed": (0, 2 ** 31)}}

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

TRANSFORM_SEMANTICS = """The boundary is a stellarator-symmetric Fourier surface,
R(t,p) = sum r_cos[m][n] cos(m t - NFP n p), Z(t,p) = sum z_sin[m][n] sin(m t - NFP n p).
r_cos[0][0] is the major radius R0. `parameters` must be numeric JSON values, EXACTLY these keys:
- structural_dilation: radial_scale (0.5-2.0), vertical_scale (0.5-2.0), mode_decay (0-1).
  Multiplies every r_cos[m][n] with (m,n) != (0,0) by 1 + (radial_scale-1) * d, and every z_sin
  the same way with vertical_scale, where d = (1-mode_decay)^(m+|n|-1). R0 is PINNED, so this
  changes the minor radius and shaping only: radial_scale > 1 grows the minor radius and therefore
  LOWERS the aspect ratio A = R0/a. mode_decay concentrates the change on the low-order modes.
- truncation_reconstruction: core_poloidal_mode (1-6), core_toroidal_mode (1-6),
  band_amplitude (1e-4 to 0.05), phase_seed (integer). Zeroes every mode outside the core
  (m <= core_poloidal_mode, |n| <= core_toroidal_mode), then REBUILDS those modes at
  +/- band_amplitude * R0 / (m+|n|) in R/Z quadrature with signs drawn from phase_seed. The
  discarded spectrum is re-derived, not copied. Large band_amplitude has been measured to break
  spectral condensation and kill the solve outright.
- mode_continuation: max_poloidal_mode (1-8), max_toroidal_mode (1-8),
  band_amplitude (1e-4 to 0.05), phase_seed (integer). Zero-pads the grid out to those limits and
  fills ONLY the newly exposed modes with the same coordinated band, leaving the existing
  spectrum untouched."""

TRANSFORM_WRITER_PROMPT = """You design ONE structural transformation of an EXISTING converged
stellarator basin, not a whole optimizer. This is the declared inherited arm: the source basin is
supplied by the host, is reported separately from independent discovery, and you never see or write
boundary coefficients. Target a 5% or larger increase in L while keeping the equilibrium solvable.
Coefficient micro-polish and gradient steps are forbidden — those are a later stage.
Allowed families: {families}.

Return ONLY one JSON object with exactly these fields:
family, version, parameters, mechanism, expected_l_gain_fraction, expected_constraint_effects,
novelty_mechanism, kill_criterion, falsifiable_prediction.
{semantics}
`expected_constraint_effects` must have EXACTLY these five keys:
aspect, iota, qi, mirror, elongation. Do not rename them or discuss other constraints there.
The evidence below reports the source basin's measured active violation. Quoting those
host-supplied numbers is expected and is NOT a provenance violation; writing raw Fourier
coefficients is. A proposal that raises L by pushing the active violation further out will be
measured as a failure, so say in `mechanism` why yours does not.
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

TRANSFORM_CRITIC_PROMPT = """Act as a hostile scientific and security reviewer. Review the proposed
structural transformation below. This is the DECLARED INHERITED ARM, so acting on a source basin is
the point and is not by itself a reason to reject, and quoting host-measured source metrics is not
a provenance violation.

The host executes these operators exactly as follows -- judge the proposal against THIS
implementation, not against an assumed one:
{semantics}

Reject it only if it is local polish, predicts <5% L gain, has no physical mechanism, repeats a
parameter region the evidence already measured as failing, lacks a cheap kill criterion, embeds raw
Fourier coefficients, or its parameters fall outside the documented ranges. If the mechanism is
plausible under the implementation above and the prediction is falsifiable, accept it: an operator
that is wrong about the physics is cheap to measure and is how this campaign learns.
Return ONLY JSON:
{{"accept": boolean, "reasons": [strings], "risk": "low|medium|high"}}

Proposal:
{proposal}

Prior evidence:
{evidence}
"""

TRANSFORM_REPAIR_PROMPT = """Your JSON proposal was rejected before physics for this exact error:
{error}
Return a corrected proposal ONLY as JSON. Preserve the physical hypothesis but obey the schema
literally. Allowed family for this round: {families}. Use exactly the documented parameter keys and
ranges for that family; do not add, rename, or drop a key.
Expected constraint keys MUST be exactly aspect,iota,qi,mirror,elongation.
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
                          str(value["kill_criterion"]),
                          independent=value["family"] not in TRANSFORM_FAMILIES)
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
        if op.family in TRANSFORM_FAMILIES:
            expected = TRANSFORM_SCHEMA[op.family]
            if set(op.parameters) != expected:
                raise ValueError(f"non-executable {op.family} parameter schema")
            for key, (low, high) in TRANSFORM_BOUNDS[op.family].items():
                if not (low <= float(op.parameters[key]) <= high):
                    raise ValueError(f"{op.family}.{key} outside [{low}, {high}]")
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

    @property
    def transform_arm(self) -> bool:
        return bool(self.allowed_families) and self.allowed_families <= TRANSFORM_FAMILIES

    @property
    def writer_prompt(self) -> str:
        """Transform arms get their own contract: they act on a source basin,
        so the independent-arm prohibition would be self-contradictory — and a
        critic told to reject inherited artifacts would refuse every one."""
        return TRANSFORM_WRITER_PROMPT if self.transform_arm else WRITER_PROMPT

    @property
    def critic_prompt(self) -> str:
        return TRANSFORM_CRITIC_PROMPT if self.transform_arm else CRITIC_PROMPT

    @property
    def repair_prompt(self) -> str:
        return TRANSFORM_REPAIR_PROMPT if self.transform_arm else REPAIR_PROMPT

    def propose(self, evidence: str) -> tuple[Proposal | None, dict[str, Any]]:
        raw = self.llm.chat(self.model, [{"role": "user", "content": self.writer_prompt.format(
            families=", ".join(sorted(self.allowed_families)), evidence=evidence[:12000],
            semantics=TRANSFORM_SEMANTICS)}],
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
                        self.repair_prompt.format(error=exc,
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
            self.critic_prompt.format(proposal=json.dumps(proposal.to_dict(), sort_keys=True),
                                 evidence=evidence[:12000],
                                 semantics=TRANSFORM_SEMANTICS)}], .2,
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
