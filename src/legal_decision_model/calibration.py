"""Conservative threshold selection and evaluation metrics."""

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from legal_decision_model.constants import NO_ATTENTION, REQUIRES_ATTENTION


@dataclass(frozen=True)
class ScoredExample:
    """A gold label and the calibrated probability assigned to safe clearance."""

    gold: str
    probability_no_attention: float


@dataclass(frozen=True)
class Metrics:
    """Metrics aligned to the cost of incorrectly clearing a matter."""

    total: int
    accuracy: float
    attention_recall: float
    false_clear_rate: float
    clearance_error_rate: float
    clearance_coverage: float
    safe_clearance_recall: float
    brier_score: float
    true_attention: int
    false_clearances: int
    true_clearances: int
    routed_for_attention: int

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-serializable metrics."""
        return asdict(self)


def probability_no_attention(logits: list[float], temperature: float) -> float:
    """Convert two logits into the calibrated probability of no attention."""
    if len(logits) != 2:
        raise ValueError(f"expected two logits, found {len(logits)}")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = [value / temperature for value in logits]
    peak = max(scaled)
    exp = [math.exp(value - peak) for value in scaled]
    return exp[0] / sum(exp)


def decide(probability_safe: float, clearance_threshold: float | None) -> str:
    """Clear only when the calibrated safe probability meets the threshold."""
    if not 0 <= probability_safe <= 1:
        raise ValueError("probability must be between 0 and 1")
    if clearance_threshold is None:
        return REQUIRES_ATTENTION
    if not 0 <= clearance_threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return NO_ATTENTION if probability_safe >= clearance_threshold else REQUIRES_ATTENTION


def select_clearance_threshold(
    examples: Iterable[ScoredExample],
    max_false_clear_rate: float = 0.0,
) -> float | None:
    """Maximize safe clearance coverage while respecting the false-clear limit."""
    rows = list(examples)
    if not rows:
        raise ValueError("cannot select a threshold without examples")
    if not 0 <= max_false_clear_rate <= 1:
        raise ValueError("max_false_clear_rate must be between 0 and 1")
    candidates = sorted(
        {0.0, 1.0, *(math.nextafter(row.probability_no_attention, 1.0) for row in rows)}
    )
    feasible: list[tuple[int, float]] = []
    for threshold in candidates:
        metrics = evaluate_scores(rows, threshold)
        if metrics.false_clear_rate <= max_false_clear_rate + 1e-12:
            feasible.append((metrics.true_clearances, threshold))
    if not feasible:
        return None
    most_true_clearances = max(item[0] for item in feasible)
    return min(threshold for count, threshold in feasible if count == most_true_clearances)


def evaluate_scores(examples: Iterable[ScoredExample], threshold: float | None) -> Metrics:
    """Evaluate conservative decisions at one threshold."""
    rows = list(examples)
    if not rows:
        raise ValueError("cannot evaluate an empty dataset")
    attention_total = sum(row.gold == REQUIRES_ATTENTION for row in rows)
    safe_total = sum(row.gold == NO_ATTENTION for row in rows)
    false_clearances = 0
    true_clearances = 0
    correct = 0
    routed = 0
    brier = 0.0
    for row in rows:
        prediction = decide(row.probability_no_attention, threshold)
        correct += prediction == row.gold
        routed += prediction == REQUIRES_ATTENTION
        false_clearances += prediction == NO_ATTENTION and row.gold == REQUIRES_ATTENTION
        true_clearances += prediction == NO_ATTENTION and row.gold == NO_ATTENTION
        target_safe = 1.0 if row.gold == NO_ATTENTION else 0.0
        brier += (row.probability_no_attention - target_safe) ** 2
    cleared = false_clearances + true_clearances
    true_attention = attention_total - false_clearances
    return Metrics(
        total=len(rows),
        accuracy=correct / len(rows),
        attention_recall=true_attention / attention_total if attention_total else 1.0,
        false_clear_rate=false_clearances / attention_total if attention_total else 0.0,
        clearance_error_rate=false_clearances / cleared if cleared else 0.0,
        clearance_coverage=cleared / len(rows),
        safe_clearance_recall=true_clearances / safe_total if safe_total else 1.0,
        brier_score=brier / len(rows),
        true_attention=true_attention,
        false_clearances=false_clearances,
        true_clearances=true_clearances,
        routed_for_attention=routed,
    )
