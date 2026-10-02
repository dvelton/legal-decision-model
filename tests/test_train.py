from pathlib import Path

from stuntd.train.layout import Layout

from legal_decision_model.constants import NO_ATTENTION, REQUIRES_ATTENTION
from legal_decision_model.records import DecisionRecord
from legal_decision_model.train import _calibrate_saved_head


class FakeDecider:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def decide(self, model, head_path: Path, text: str):
        assert head_path == Path("serialized-head.safetensors")
        safe_probability = 0.9 if text == "safe" else 0.8
        return type(
            "Verdict",
            (),
            {"probabilities": (safe_probability, 1.0 - safe_probability)},
        )()


def _record(record_id: str, text: str, label: str) -> DecisionRecord:
    return DecisionRecord(
        record_id=record_id,
        pair_id="pair",
        split="validation",
        family="contracts",
        style="ticket",
        template_id=record_id,
        text=text,
        facts={},
        label=label,
        reasons=(),
    )


def test_calibration_uses_serialized_serving_probabilities(monkeypatch) -> None:
    monkeypatch.setattr("legal_decision_model.train.Decider", FakeDecider)
    monkeypatch.setattr("legal_decision_model.train.fit_temperature", lambda logits, labels: 1.0)
    temperature, threshold, metrics = _calibrate_saved_head(
        Path("serialized-head.safetensors"),
        [
            _record("safe", "safe", NO_ATTENTION),
            _record("attention", "attention", REQUIRES_ATTENTION),
        ],
        Layout(512, 192, True),
        device="cpu",
        training_rows=2,
        max_false_clear_rate=0.0,
    )
    assert temperature == 1.0
    assert threshold is not None and threshold > 0.8
    assert metrics["false_clearances"] == 0
    assert metrics["true_clearances"] == 1
