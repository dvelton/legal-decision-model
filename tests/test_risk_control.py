import math

from legal_decision_model.constants import NO_ATTENTION, REQUIRES_ATTENTION
from legal_decision_model.risk_control import (
    CLEARANCE_NUMERICAL_MARGIN,
    RiskExample,
    exact_binomial_upper_bound,
    select_risk_controlled_threshold,
)


def _rows(values: list[tuple[str, float]]) -> list[RiskExample]:
    return [
        RiskExample(label, score, f"trial-{index}") for index, (label, score) in enumerate(values)
    ]


def test_exact_binomial_upper_bound_is_one_without_evidence() -> None:
    assert exact_binomial_upper_bound(0, 0, 0.95) == 1.0


def test_exact_binomial_upper_bound_matches_zero_error_formula() -> None:
    assert exact_binomial_upper_bound(0, 1_000, 0.95) < exact_binomial_upper_bound(0, 100, 0.95)
    assert math.isclose(
        exact_binomial_upper_bound(0, 3_000, 0.95),
        1 - 0.05 ** (1 / 3_000),
    )
    assert math.isclose(exact_binomial_upper_bound(100, 100, 0.95), 1.0)


def test_exact_binomial_bound_rejects_nominal_wilson_edge_case() -> None:
    assert exact_binomial_upper_bound(0, 270, 0.95) > 0.01


def test_risk_control_maximizes_true_clearances() -> None:
    rows = _rows(
        [
            (NO_ATTENTION, 0.95),
            (NO_ATTENTION, 0.85),
            *[(REQUIRES_ATTENTION, 0.4 - index / 1_000) for index in range(100)],
        ]
    )
    selection = select_risk_controlled_threshold(
        rows,
        maximum_false_clear_rate=0.05,
        confidence_level=0.95,
    )
    assert selection.threshold is not None
    assert selection.point.true_clearances == 2
    assert selection.point.false_clearances == 0
    assert selection.point.upper_false_clear_rate <= 0.05


def test_risk_control_disables_clearance_without_enough_evidence() -> None:
    rows = _rows(
        [
            (NO_ATTENTION, 0.95),
            *[(REQUIRES_ATTENTION, 0.1) for _ in range(10)],
        ]
    )
    selection = select_risk_controlled_threshold(
        rows,
        maximum_false_clear_rate=0.001,
        confidence_level=0.95,
    )
    assert selection.threshold is None
    assert selection.point.cleared == 0


def test_risk_control_can_require_zero_observed_false_clearances() -> None:
    rows = _rows(
        [
            (NO_ATTENTION, 0.99),
            (NO_ATTENTION, 0.95),
            (REQUIRES_ATTENTION, 0.97),
            *[(REQUIRES_ATTENTION, 0.1) for _ in range(5_000)],
        ]
    )
    selection = select_risk_controlled_threshold(
        rows,
        maximum_false_clear_rate=0.001,
        maximum_false_clearances=0,
        confidence_level=0.95,
    )
    assert selection.point.false_clearances == 0
    assert selection.point.true_clearances == 1


def test_threshold_includes_float32_numerical_margin() -> None:
    risky_score = 0.999_994_726_847_944_5
    rows = _rows(
        [
            (NO_ATTENTION, 0.999_999),
            (REQUIRES_ATTENTION, risky_score),
            *[(REQUIRES_ATTENTION, 0.1) for _ in range(5_000)],
        ]
    )
    selection = select_risk_controlled_threshold(
        rows,
        maximum_false_clear_rate=0.001,
        maximum_false_clearances=0,
        confidence_level=0.95,
    )
    assert selection.threshold is not None
    assert selection.threshold > risky_score + CLEARANCE_NUMERICAL_MARGIN


def test_safe_candidate_cannot_bypass_risky_numerical_margin() -> None:
    risky_score = 0.8
    rows = _rows(
        [
            (NO_ATTENTION, risky_score + CLEARANCE_NUMERICAL_MARGIN / 2),
            (NO_ATTENTION, risky_score + CLEARANCE_NUMERICAL_MARGIN * 2),
            (REQUIRES_ATTENTION, risky_score),
            *[(REQUIRES_ATTENTION, 0.1) for _ in range(5_000)],
        ]
    )
    selection = select_risk_controlled_threshold(
        rows,
        maximum_false_clear_rate=0.001,
        maximum_false_clearances=0,
        confidence_level=0.95,
    )
    assert selection.threshold is not None
    assert selection.threshold > risky_score + CLEARANCE_NUMERICAL_MARGIN
    assert selection.point.true_clearances == 1


def test_correlated_renderings_count_as_one_attention_trial() -> None:
    rows = [
        RiskExample(REQUIRES_ATTENTION, 0.1, f"scenario-{index // 2}") for index in range(5_000)
    ]
    selection = select_risk_controlled_threshold(
        rows,
        maximum_false_clear_rate=0.001,
        maximum_false_clearances=0,
        confidence_level=0.95,
    )
    assert selection.threshold is None
    assert selection.point.attention_examples == 5_000
    assert selection.point.attention_trials == 2_500
    assert selection.point.upper_false_clear_rate > 0.001
