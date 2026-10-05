"""Finite-sample risk-controlled threshold selection."""

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import cache

from legal_decision_model.constants import NO_ATTENTION, REQUIRES_ATTENTION

CLEARANCE_NUMERICAL_MARGIN = 1e-6


@dataclass(frozen=True)
class RiskExample:
    """One scored record and its independent calibration-trial identity."""

    label: str
    score: float
    trial_id: str


@dataclass(frozen=True)
class RiskPoint:
    """One threshold on the safe-score risk/coverage curve."""

    threshold: float | None
    total: int
    cleared: int
    true_clearances: int
    false_clearances: int
    attention_examples: int
    false_clearance_trials: int
    attention_trials: int
    observed_false_clear_rate: float
    upper_false_clear_rate: float
    clearance_coverage: float

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class RiskSelection:
    """Selected operating point and its supporting curve."""

    threshold: float | None
    confidence_level: float
    maximum_false_clear_rate: float
    maximum_false_clearances: int | None
    point: RiskPoint
    curve: tuple[RiskPoint, ...]


def _binomial_cdf(errors: int, trials: int, probability: float) -> float:
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 1.0 if errors == trials else 0.0
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    log_terms = [
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        + successes * log_probability
        + (trials - successes) * log_complement
        for successes in range(errors + 1)
    ]
    largest = max(log_terms)
    return math.exp(largest) * math.fsum(math.exp(log_term - largest) for log_term in log_terms)


@cache
def exact_binomial_upper_bound(
    errors: int,
    trials: int,
    confidence_level: float,
) -> float:
    """Exact one-sided Clopper-Pearson upper bound for a binomial error rate."""
    if trials < 0 or errors < 0 or errors > trials:
        raise ValueError("errors and trials must satisfy 0 <= errors <= trials")
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0.5 and 1")
    if trials == 0 or errors == trials:
        return 1.0
    alpha = 1 - confidence_level
    if errors == 0:
        return -math.expm1(math.log(alpha) / trials)
    lower = errors / trials
    upper = 1.0
    for _ in range(64):
        midpoint = (lower + upper) / 2
        if _binomial_cdf(errors, trials, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _point(
    rows: list[RiskExample],
    threshold: float | None,
    confidence_level: float,
) -> RiskPoint:
    if threshold is None:
        false_clearances = 0
        true_clearances = 0
        false_clearance_trial_ids: set[str] = set()
    else:
        false_clearances = sum(
            row.label == REQUIRES_ATTENTION
            and min(1.0, row.score + CLEARANCE_NUMERICAL_MARGIN) >= threshold
            for row in rows
        )
        false_clearance_trial_ids = {
            row.trial_id
            for row in rows
            if row.label == REQUIRES_ATTENTION
            and min(1.0, row.score + CLEARANCE_NUMERICAL_MARGIN) >= threshold
        }
        true_clearances = sum(row.label == NO_ATTENTION and row.score >= threshold for row in rows)
    cleared = false_clearances + true_clearances
    attention_examples = sum(row.label == REQUIRES_ATTENTION for row in rows)
    attention_trial_ids = {row.trial_id for row in rows if row.label == REQUIRES_ATTENTION}
    false_clearance_trials = len(false_clearance_trial_ids)
    attention_trials = len(attention_trial_ids)
    observed = false_clearance_trials / attention_trials if attention_trials else 0.0
    return RiskPoint(
        threshold=threshold,
        total=len(rows),
        cleared=cleared,
        true_clearances=true_clearances,
        false_clearances=false_clearances,
        attention_examples=attention_examples,
        false_clearance_trials=false_clearance_trials,
        attention_trials=attention_trials,
        observed_false_clear_rate=observed,
        upper_false_clear_rate=exact_binomial_upper_bound(
            false_clearance_trials,
            attention_trials,
            confidence_level,
        ),
        clearance_coverage=cleared / len(rows) if rows else 0.0,
    )


def select_risk_controlled_threshold(
    examples: Iterable[RiskExample],
    *,
    maximum_false_clear_rate: float,
    confidence_level: float,
    maximum_false_clearances: int | None = None,
) -> RiskSelection:
    """Maximize true clearances subject to a one-sided false-clear upper bound."""
    rows = list(examples)
    if not rows:
        raise ValueError("risk control needs at least one example")
    if not 0 <= maximum_false_clear_rate <= 1:
        raise ValueError("maximum_false_clear_rate must be between 0 and 1")
    if maximum_false_clearances is not None and maximum_false_clearances < 0:
        raise ValueError("maximum_false_clearances must be non-negative")
    if any(not row.trial_id for row in rows):
        raise ValueError("risk-control trial IDs must not be empty")
    candidates = sorted(
        {
            row.score
            for row in rows
            if row.label == NO_ATTENTION and math.isfinite(row.score) and 0 <= row.score <= 1
        }
        | {
            math.nextafter(row.score + CLEARANCE_NUMERICAL_MARGIN, 1.0)
            for row in rows
            if row.label == REQUIRES_ATTENTION
            and math.isfinite(row.score)
            and 0 <= row.score <= 1
            and row.score + CLEARANCE_NUMERICAL_MARGIN < 1
        }
    )
    curve = tuple(_point(rows, threshold, confidence_level) for threshold in (*candidates, None))
    feasible = [
        point
        for point in curve
        if point.upper_false_clear_rate <= maximum_false_clear_rate
        and (
            maximum_false_clearances is None
            or point.false_clearance_trials <= maximum_false_clearances
        )
    ]
    if not feasible:
        disabled = _point(rows, None, confidence_level)
        return RiskSelection(
            None,
            confidence_level,
            maximum_false_clear_rate,
            maximum_false_clearances,
            disabled,
            curve,
        )
    selected = max(
        feasible,
        key=lambda point: (
            point.true_clearances,
            -point.false_clearance_trials,
            -point.false_clearances,
            point.clearance_coverage,
            float("-inf") if point.threshold is None else -point.threshold,
        ),
    )
    return RiskSelection(
        selected.threshold,
        confidence_level,
        maximum_false_clear_rate,
        maximum_false_clearances,
        selected,
        curve,
    )
