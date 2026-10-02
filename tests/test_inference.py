from pathlib import Path
from types import SimpleNamespace

from legal_decision_model.constants import REQUIRES_ATTENTION
from legal_decision_model.inference import LegalDecisionModel


class FakeTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text: str, **kwargs):
        tokens = list(range(10, 10 + len(text.split())))
        maximum = kwargs.get("max_length")
        if kwargs.get("truncation") and maximum is not None:
            tokens = tokens[:maximum]
        return {"input_ids": tokens}


class FailingDecider:
    def decide(self, *args, **kwargs):
        raise AssertionError("truncated input must not reach the model")


def test_over_budget_input_routes_to_attention() -> None:
    model = LegalDecisionModel.__new__(LegalDecisionModel)
    model.metadata = SimpleNamespace(
        max_len=24,
        head_max_len=12,
        clearance_threshold=0.8,
        base_model="test-model",
        policy_version="northstar-1.0",
    )
    model._tokenizer = FakeTokenizer()
    model._question = {
        "t": "choice",
        "ins": "requires attention",
        "crit": {
            "NO_HUMAN_LAWYER_ATTENTION": None,
            "REQUIRES_HUMAN_LAWYER_ATTENTION": None,
        },
    }
    model._decider = FailingDecider()
    model._site_model = object()
    model.head_path = Path("unused")

    result = model.predict("word " * 100)

    assert result.decision == REQUIRES_ATTENTION
    assert result.routing_reason == "INPUT_EXCEEDS_MODEL_TOKEN_BUDGET"
    assert result.probability_no_human_lawyer_attention == 0.0
