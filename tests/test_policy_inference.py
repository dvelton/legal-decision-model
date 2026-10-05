from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from legal_decision_model.policy_inference import PolicyDecisionModel
from legal_decision_model.policy_model import FINAL_TASK, IN_SCOPE_TASK
from legal_decision_model.policy_schema import load_policy, policy_fingerprint


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2


class FakeEncoder:
    tokenizer = FakeTokenizer()

    def decode_tokens(self, ids: list[int]) -> str:
        return " ".join(map(str, ids))


def _model() -> PolicyDecisionModel:
    model = PolicyDecisionModel.__new__(PolicyDecisionModel)
    model.policy = load_policy()
    model.metadata = SimpleNamespace(
        max_tokens=6,
        window_overlap=2,
        maximum_windows=3,
        task_names=(
            *model.policy.prerequisites,
            *model.policy.triggers,
            IN_SCOPE_TASK,
            FINAL_TASK,
        ),
        global_threshold=0.8,
        family_thresholds={"contracts": 0.85},
        base_model="test-model",
        policy_version="test-policy",
    )
    model.encoder = FakeEncoder()
    return model


def test_long_input_is_split_into_overlapping_windows() -> None:
    model = _model()
    assert model._windows("original", list(range(8))) == [
        "0 1 2 3",
        "2 3 4 5",
        "4 5 6 7",
    ]


def test_excessive_windows_disable_model_inference() -> None:
    model = _model()
    assert model._windows("original", list(range(10))) is None


def test_window_aggregation_uses_most_conservative_signal() -> None:
    model = _model()
    tasks = model.metadata.task_names
    probabilities = torch.full((2, len(tasks)), 0.1)
    probabilities[:, tasks.index(model.policy.prerequisites[0])] = torch.tensor([0.95, 0.7])
    probabilities[:, tasks.index(model.policy.triggers[0])] = torch.tensor([0.05, 0.8])
    probabilities[:, tasks.index(IN_SCOPE_TASK)] = torch.tensor([0.99, 0.75])
    probabilities[:, tasks.index(FINAL_TASK)] = torch.tensor([0.1, 0.9])
    aggregated = model._aggregate_tasks(probabilities)
    assert aggregated[model.policy.prerequisites[0]] == pytest.approx(0.7)
    assert aggregated[model.policy.triggers[0]] == pytest.approx(0.8)
    assert aggregated[IN_SCOPE_TASK] == pytest.approx(0.75)
    assert aggregated[FINAL_TASK] == pytest.approx(0.9)


def test_family_threshold_cannot_weaken_global_threshold() -> None:
    model = _model()
    assert model._threshold("contracts") == 0.85


def test_non_finite_features_route_conservatively() -> None:
    model = _model()
    decision = model._decision_from_features(
        torch.tensor([[float("nan")]]),
        token_count=1,
        window_count=1,
        started=0.0,
    )
    assert decision.requires_human_lawyer_attention
    assert decision.routing_reason == "NON_FINITE_MODEL_OUTPUT"


def test_single_member_disagreement_is_zero() -> None:
    probabilities = torch.tensor([[[0.25, 0.75]]])
    assert probabilities.std(dim=0, correction=0).max() == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("confidence_level", 0.99),
        ("maximum_false_clear_rate", 0.0001),
        ("maximum_group_false_clear_rate", 0.001),
        ("minimum_group_attention_examples", 500),
    ),
)
def test_artifact_rejects_changed_risk_policy(field: str, value: float | int) -> None:
    original = load_policy()
    changed = replace(
        original,
        risk_control=replace(original.risk_control, **{field: value}),
    )
    model = PolicyDecisionModel.__new__(PolicyDecisionModel)
    model.policy = changed
    model.metadata = SimpleNamespace(
        format_version=3,
        policy_name=original.name,
        policy_version=original.version,
        policy_fingerprint=policy_fingerprint(original),
    )
    with pytest.raises(ValueError, match="policy settings"):
        model._validate_artifact()
