"""Validate fictional human-authored behavioral test contributions."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_decision_model.generate import FAMILIES
from legal_decision_model.policy import evaluate_facts
from legal_decision_model.policy_schema import load_policy
from legal_decision_model.validate import FORBIDDEN_DATA_PATTERNS

CASE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
REVIEW_STATES = {"pending", "accepted", "rejected"}


@dataclass(frozen=True)
class ContributionValidation:
    """Validated contribution summary."""

    case_id: str
    policy_version: str
    family: str
    capabilities: tuple[str, ...]
    changed_facts: tuple[str, ...]
    review_status: str
    human_reviewed: bool


def _mapping(raw: object, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{where} must be an object with string keys")
    return dict(raw)


def _strings(raw: object, where: str, *, exact_length: int | None = None) -> tuple[str, ...]:
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(value, str) and value.strip() for value in raw)
    ):
        raise ValueError(f"{where} must be a non-empty string array")
    values = tuple(value.strip() for value in raw)
    if exact_length is not None and len(values) != exact_length:
        raise ValueError(f"{where} must contain exactly {exact_length} values")
    if len(set(values)) != len(values):
        raise ValueError(f"{where} must not contain duplicates")
    return values


def _record(raw: object, where: str) -> tuple[dict[str, bool], tuple[str, str]]:
    record = _mapping(raw, where)
    facts = load_policy().validate_facts(_mapping(record.get("facts"), f"{where}.facts"))
    renderings = _strings(
        record.get("renderings"),
        f"{where}.renderings",
        exact_length=2,
    )
    for rendering in renderings:
        if len(rendering) < 20:
            raise ValueError(f"{where}.renderings must contain substantive fictional text")
        for name, pattern in FORBIDDEN_DATA_PATTERNS.items():
            if pattern.search(rendering):
                raise ValueError(f"{where}.renderings matches forbidden {name}")
    return facts, (renderings[0], renderings[1])


def validate_contribution(path: Path) -> ContributionValidation:
    """Validate one fictional contribution file against the active policy."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    value = _mapping(raw, str(path))
    policy = load_policy()
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
        raise ValueError("case_id must contain 3-80 lowercase letters, digits, or hyphens")
    if value.get("fictional_original_content") is not True:
        raise ValueError("fictional_original_content must be true")
    if value.get("policy_version") != policy.version:
        raise ValueError(f"contribution must target policy version {policy.version}")
    family = value.get("family")
    families = {spec.name for spec in FAMILIES}
    if not isinstance(family, str) or family not in families:
        raise ValueError(f"family must be one of: {', '.join(sorted(families))}")
    capabilities = _strings(value.get("capabilities"), "capabilities")
    changed_facts = _strings(value.get("changed_facts"), "changed_facts")
    if not set(changed_facts) <= set(policy.fact_names):
        raise ValueError("changed_facts contains an unknown policy fact")
    safe_facts, _ = _record(value.get("safe"), "safe")
    attention_facts, _ = _record(value.get("attention"), "attention")
    actual_changed = {
        name for name in policy.fact_names if safe_facts[name] != attention_facts[name]
    }
    if actual_changed != set(changed_facts):
        raise ValueError("changed_facts does not match the two structured fact sets")
    if evaluate_facts(safe_facts, policy).requires_human_lawyer_attention:
        raise ValueError("safe facts do not produce the no-attention label")
    if not evaluate_facts(attention_facts, policy).requires_human_lawyer_attention:
        raise ValueError("attention facts do not produce the attention-required label")
    review = _mapping(value.get("review"), "review")
    status = review.get("status")
    if not isinstance(status, str) or status not in REVIEW_STATES:
        raise ValueError("review.status must be pending, accepted, or rejected")
    human_reviewed = status == "accepted"
    if human_reviewed and (
        not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not isinstance(review.get("reviewed_at"), str)
        or not review["reviewed_at"].strip()
    ):
        raise ValueError("accepted contributions require reviewer and reviewed_at")
    return ContributionValidation(
        case_id=case_id,
        policy_version=policy.version,
        family=family,
        capabilities=capabilities,
        changed_facts=changed_facts,
        review_status=status,
        human_reviewed=human_reviewed,
    )
