import pytest

from legal_decision_model.constants import NO_ATTENTION
from legal_decision_model.policy import ScenarioFacts
from legal_decision_model.policy_schema import load_policy
from legal_decision_model.records import DecisionRecord
from legal_decision_model.validate import _validate_record


def test_v2_validation_rejects_missing_structured_fact() -> None:
    facts = ScenarioFacts().to_dict()
    del facts["security_incident"]
    record = DecisionRecord.from_dict(
        {
            "record_id": "missing-fact",
            "pair_id": "pair",
            "split": "test",
            "family": "contracts",
            "style": "ticket",
            "template_id": "template",
            "text": "Fictional request.",
            "facts": facts,
            "label": NO_ATTENTION,
            "reasons": ["APPROVED_SELF_SERVICE_PROCESS"],
            "scenario_id": "scenario",
            "rendering_id": "rendering",
        }
    )
    errors = _validate_record(record, require_v2_metadata=True)
    assert any("facts must exactly match" in error for error in errors)


def test_record_parser_rejects_non_boolean_fact() -> None:
    facts = ScenarioFacts().to_dict()
    facts["security_incident"] = "no"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="boolean"):
        DecisionRecord.from_dict(
            {
                "record_id": "bad-fact",
                "pair_id": "pair",
                "split": "test",
                "family": "contracts",
                "style": "ticket",
                "template_id": "template",
                "text": "Fictional request.",
                "facts": facts,
                "label": NO_ATTENTION,
                "reasons": ["APPROVED_SELF_SERVICE_PROCESS"],
            }
        )


def test_record_parser_accepts_custom_boolean_fact_schema() -> None:
    policy = load_policy()
    raw = {
        "record_id": "custom-facts",
        "pair_id": "pair",
        "split": "test",
        "family": "contracts",
        "style": "ticket",
        "template_id": "template",
        "text": "Fictional request.",
        "facts": {"custom_trigger": False},
        "label": policy.no_attention_label,
        "reasons": [policy.success_reason],
    }
    assert DecisionRecord.from_dict(raw).facts == {"custom_trigger": False}
