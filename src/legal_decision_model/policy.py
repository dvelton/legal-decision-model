"""Executable fictional policy that supplies synthetic ground-truth labels."""

from dataclasses import dataclass, fields
from typing import Any

from legal_decision_model.constants import DEFAULT_POLICY_PATH
from legal_decision_model.policy_schema import PolicySpec, load_policy

DEFAULT_POLICY = load_policy(DEFAULT_POLICY_PATH)


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


TRIGGER_REASONS = DEFAULT_POLICY.trigger_reasons


@dataclass(frozen=True)
class PolicyDecision:
    """One deterministic policy decision."""

    label: str
    requires_human_lawyer_attention: bool
    reasons: tuple[str, ...]
    policy_version: str = DEFAULT_POLICY.version


def evaluate_facts(
    facts: dict[str, bool],
    policy: PolicySpec = DEFAULT_POLICY,
) -> PolicyDecision:
    """Apply a compiled policy to a structured fact mapping."""
    validated = policy.validate_facts(facts)
    reasons = [policy.trigger_reasons[name] for name in policy.triggers if validated[name]]
    reasons.extend(
        policy.prerequisite_reasons[name] for name in policy.prerequisites if not validated[name]
    )
    if reasons:
        return PolicyDecision(
            policy.requires_attention_label,
            True,
            tuple(reasons),
            policy.version,
        )
    return PolicyDecision(
        policy.no_attention_label,
        False,
        (policy.success_reason,),
        policy.version,
    )


def evaluate_policy(facts: ScenarioFacts) -> PolicyDecision:
    """Apply the fictional policy conservatively."""
    return evaluate_facts(facts.to_dict())
