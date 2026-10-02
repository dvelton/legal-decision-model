"""Executable fictional policy that supplies synthetic ground-truth labels."""

from dataclasses import dataclass, fields
from typing import Any

from legal_decision_model.constants import (
    NO_ATTENTION,
    POLICY_VERSION,
    REQUIRES_ATTENTION,
)


@dataclass(frozen=True)
class ScenarioFacts:
    """Facts evaluated by the fictional Northstar Legal Attention Policy."""

    approved_self_service_process: bool = True
    legal_interpretation: bool = False
    nonstandard_contract_terms: bool = False
    dispute_or_claim: bool = False
    regulator_or_government_contact: bool = False
    sensitive_personal_data: bool = False
    security_incident: bool = False
    employment_action: bool = False
    intellectual_property_ownership: bool = False
    new_jurisdiction: bool = False
    material_external_commitment: bool = False
    incomplete_or_conflicting_facts: bool = False
    outside_approved_playbook: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ScenarioFacts":
        """Build facts while refusing unknown or non-boolean fields."""
        expected = {item.name for item in fields(cls)}
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown policy facts: {', '.join(sorted(unknown))}")
        if any(not isinstance(value, bool) for value in raw.values()):
            raise ValueError("every policy fact must be a boolean")
        return cls(**raw)

    def to_dict(self) -> dict[str, bool]:
        """Return JSON-serializable facts."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


TRIGGER_REASONS = {
    "legal_interpretation": "The request asks for individualized legal interpretation.",
    "nonstandard_contract_terms": "The request includes nonstandard contract terms.",
    "dispute_or_claim": "The request involves a dispute, claim, or threatened claim.",
    "regulator_or_government_contact": "A regulator or government authority is involved.",
    "sensitive_personal_data": "The request involves sensitive personal data.",
    "security_incident": "The request involves a security incident or suspected compromise.",
    "employment_action": "The request involves an individualized employment action.",
    "intellectual_property_ownership": "The request raises intellectual-property ownership.",
    "new_jurisdiction": "The request involves a jurisdiction outside the approved playbook.",
    "material_external_commitment": "The request would create a material external commitment.",
    "incomplete_or_conflicting_facts": "The supplied facts are incomplete or conflicting.",
    "outside_approved_playbook": "The request falls outside an approved self-service playbook.",
}


@dataclass(frozen=True)
class PolicyDecision:
    """One deterministic policy decision."""

    label: str
    requires_human_lawyer_attention: bool
    reasons: tuple[str, ...]
    policy_version: str = POLICY_VERSION


def evaluate_policy(facts: ScenarioFacts) -> PolicyDecision:
    """Apply the fictional policy conservatively."""
    reasons = [
        reason for field_name, reason in TRIGGER_REASONS.items() if getattr(facts, field_name)
    ]
    if not facts.approved_self_service_process:
        reasons.append("No approved self-service process fully resolves the request.")
    if reasons:
        return PolicyDecision(REQUIRES_ATTENTION, True, tuple(reasons))
    return PolicyDecision(
        NO_ATTENTION,
        False,
        ("An approved self-service process fully resolves the complete request.",),
    )
