"""Evaluate the trained model on held-out synthetic data."""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from legal_decision_model.calibration import ScoredExample, evaluate_scores
from legal_decision_model.constants import DATA_DIR, MODEL_DIR, PROJECT_ROOT
from legal_decision_model.inference import LegalDecisionModel
from legal_decision_model.records import read_jsonl


@dataclass(frozen=True)
class EvaluationReport:
    """Held-out metrics and counterfactual-pair consistency."""

    split: str
    metrics: dict[str, int | float]
    counterfactual_pair_consistency: float
    family_metrics: dict[str, dict[str, int | float]]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable evaluation output."""
        return asdict(self)


def evaluate_split(
    split: str,
    data_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
    output_dir: Path = PROJECT_ROOT / "evaluation" / "current",
    device: str = "auto",
) -> EvaluationReport:
    """Run one held-out split through the persisted model."""
    records = read_jsonl(data_dir / f"{split}.jsonl")
    model = LegalDecisionModel(model_dir, device=device)
    predictions = []
    by_family: dict[str, list[ScoredExample]] = defaultdict(list)
    by_pair: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        result = model.predict(record.text)
        scored = ScoredExample(record.label, result.probability_no_human_lawyer_attention)
        predictions.append(scored)
        by_family[record.family].append(scored)
        by_pair[record.pair_id].append((record.label, result.decision))
    consistent = sum(
        len(pair) == 2
        and len({gold for gold, _ in pair}) == 2
        and all(gold == predicted for gold, predicted in pair)
        for pair in by_pair.values()
    )
    report = EvaluationReport(
        split=split,
        metrics=evaluate_scores(predictions, model.metadata.clearance_threshold).to_dict(),
        counterfactual_pair_consistency=consistent / len(by_pair) if by_pair else 0.0,
        family_metrics={
            family: evaluate_scores(rows, model.metadata.clearance_threshold).to_dict()
            for family, rows in sorted(by_family.items())
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split}.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
