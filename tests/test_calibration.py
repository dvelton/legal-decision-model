import math

from legal_decision_model.calibration import (
    ScoredExample,
    decide,
    evaluate_scores,
    probability_no_attention,
    select_clearance_threshold,
)
from legal_decision_model.constants import NO_ATTENTION, REQUIRES_ATTENTION


def test_probability_is_stable() -> None:
    probability = probability_no_attention([1001.0, 1000.0], 1.0)
    assert math.isclose(probability, 0.7310585786)


def test_zero_false_clear_threshold() -> None:
    rows = [
        ScoredExample(NO_ATTENTION, 0.99),
        ScoredExample(NO_ATTENTION, 0.80),
        ScoredExample(REQUIRES_ATTENTION, 0.79),
        ScoredExample(REQUIRES_ATTENTION, 0.10),
    ]
    threshold = select_clearance_threshold(rows, max_false_clear_rate=0.0)
    metrics = evaluate_scores(rows, threshold)
    assert metrics.false_clearances == 0
    assert metrics.true_clearances == 2
    assert threshold > 0.79
    assert decide(0.79, threshold) == REQUIRES_ATTENTION
    assert decide(0.99, threshold) == NO_ATTENTION


def test_infeasible_threshold_disables_clearance() -> None:
    rows = [
        ScoredExample(REQUIRES_ATTENTION, 1.0),
        ScoredExample(NO_ATTENTION, 1.0),
    ]
    threshold = select_clearance_threshold(rows, max_false_clear_rate=0.0)
    metrics = evaluate_scores(rows, threshold)
    assert threshold is None
    assert decide(1.0, threshold) == REQUIRES_ATTENTION
    assert metrics.false_clearances == 0
    assert metrics.clearance_coverage == 0.0


def test_evaluation_metrics_center_false_clearances() -> None:
    metrics = evaluate_scores(
        [
            ScoredExample(REQUIRES_ATTENTION, 0.9),
            ScoredExample(REQUIRES_ATTENTION, 0.1),
            ScoredExample(NO_ATTENTION, 0.8),
            ScoredExample(NO_ATTENTION, 0.2),
        ],
        threshold=0.5,
    )
    assert metrics.false_clear_rate == 0.5
    assert metrics.clearance_error_rate == 0.5
    assert metrics.attention_recall == 0.5
