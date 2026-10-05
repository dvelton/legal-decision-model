import json
from pathlib import Path

import pytest

from legal_decision_model.contribution import validate_contribution
from legal_decision_model.policy_schema import load_policy


def _case() -> dict[str, object]:
    policy = load_policy()
    safe = policy.default_facts()
    attention = dict(safe)
    attention[policy.triggers[0]] = True
    return {
        "case_id": "fictional-contract-change",
        "policy_version": policy.version,
        "fictional_original_content": True,
        "family": "contracts",
        "capabilities": ["indirect_trigger"],
        "changed_facts": [policy.triggers[0]],
        "safe": {
            "facts": safe,
            "renderings": [
                "The fictional standard request remains inside the approved process.",
                "The approved fictional workflow fully resolves this unchanged request.",
            ],
        },
        "attention": {
            "facts": attention,
            "renderings": [
                "The fictional request now requires an individualized legal interpretation.",
                "The approved workflow has no answer for this fictional legal interpretation.",
            ],
        },
        "review": {"status": "pending"},
    }


def test_valid_fictional_contribution(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    path.write_text(json.dumps(_case()), encoding="utf-8")
    result = validate_contribution(path)
    assert result.review_status == "pending"
    assert result.human_reviewed is False


def test_accepted_contribution_requires_review_metadata(tmp_path: Path) -> None:
    value = _case()
    value["review"] = {"status": "accepted"}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer and reviewed_at"):
        validate_contribution(path)
