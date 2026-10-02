from dataclasses import replace

import pytest

from legal_decision_model.constants import NO_ATTENTION, REQUIRES_ATTENTION
from legal_decision_model.policy import (
    TRIGGER_REASONS,
    ScenarioFacts,
    evaluate_policy,
)


def test_approved_complete_request_clears() -> None:
    decision = evaluate_policy(ScenarioFacts())
    assert decision.label == NO_ATTENTION
    assert decision.requires_human_lawyer_attention is False


@pytest.mark.parametrize("field_name", sorted(TRIGGER_REASONS))
def test_every_trigger_requires_attention(field_name: str) -> None:
    facts = replace(ScenarioFacts(), **{field_name: True})
    decision = evaluate_policy(facts)
    assert decision.label == REQUIRES_ATTENTION
    assert decision.requires_human_lawyer_attention is True
    assert TRIGGER_REASONS[field_name] in decision.reasons


def test_missing_self_service_process_requires_attention() -> None:
    decision = evaluate_policy(ScenarioFacts(approved_self_service_process=False))
    assert decision.label == REQUIRES_ATTENTION


def test_unknown_fact_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown policy facts"):
        ScenarioFacts.from_dict({"unknown": True})
