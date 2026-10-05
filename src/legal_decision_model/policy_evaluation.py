"""Behavioral evaluation for the multi-task policy model."""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from legal_decision_model.constants import (
    BENCHMARK_DIR,
    POLICY_MODEL_DIR,
    PROJECT_ROOT,
)
from legal_decision_model.policy_inference import PolicyDecision, PolicyDecisionModel
from legal_decision_model.records import DecisionRecord, read_jsonl
from legal_decision_model.risk_control import exact_binomial_upper_bound


@dataclass(frozen=True)
class PolicyMetrics:
    """Risk-first binary routing metrics."""

    total: int
    attention_examples: int
    no_attention_examples: int
    false_clearances: int
    false_clearance_scenarios: int
    attention_scenarios: int
    true_clearances: int
    attention_recall: float
    clearance_coverage: float
    clearance_error_rate: float
    observed_false_clear_rate: float
    upper_false_clear_rate: float
    accuracy: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyEvaluationReport:
    """One evaluated benchmark split."""

    split: str
    artifact_fingerprint: str
    policy_version: str
    base_model: str
    base_model_revision: str
    benchmark_split_sha256: str
    benchmark_manifest_sha256: str | None
    benchmark_seed: int | None
    metrics: PolicyMetrics
    counterfactual_pair_consistency: float
    paraphrase_invariance: float
    safe_score_monotonicity: float
    out_of_distribution_attention_rate: float
    family_metrics: dict[str, PolicyMetrics]
    generator_metrics: dict[str, PolicyMetrics]
    capability_metrics: dict[str, PolicyMetrics]
    routing_reasons: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _benchmark_identity(
    split: str,
    split_path: Path,
) -> tuple[str, str | None, int | None]:
    split_sha256 = _sha256(split_path)
    manifest_path = PROJECT_ROOT / "evaluation" / "policy-v2" / "benchmark-manifest.json"
    if not manifest_path.is_file():
        return split_sha256, None, None
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path}: expected an object")
    split_hashes = raw.get("split_sha256")
    if not isinstance(split_hashes, dict) or split_hashes.get(split) != split_sha256:
        return split_sha256, None, None
    seed = raw.get("seed")
    if not isinstance(seed, int):
        raise ValueError(f"{manifest_path}: seed must be an integer")
    return split_sha256, _sha256(manifest_path), seed


def _metrics(
    records: list[DecisionRecord],
    decisions: list[PolicyDecision],
    confidence_level: float,
) -> PolicyMetrics:
    if len(records) != len(decisions):
        raise ValueError("records and decisions must have the same length")
    attention = [
        index for index, record in enumerate(records) if record.requires_human_lawyer_attention
    ]
    no_attention = [
        index for index, record in enumerate(records) if not record.requires_human_lawyer_attention
    ]
    false_clearances = sum(
        not decisions[index].requires_human_lawyer_attention for index in attention
    )
    attention_scenario_ids = {records[index].scenario_id for index in attention}
    false_clearance_scenario_ids = {
        records[index].scenario_id
        for index in attention
        if not decisions[index].requires_human_lawyer_attention
    }
    true_clearances = sum(
        not decisions[index].requires_human_lawyer_attention for index in no_attention
    )
    cleared = false_clearances + true_clearances
    correct = sum(
        decision.requires_human_lawyer_attention == record.requires_human_lawyer_attention
        for record, decision in zip(records, decisions, strict=True)
    )
    return PolicyMetrics(
        total=len(records),
        attention_examples=len(attention),
        no_attention_examples=len(no_attention),
        false_clearances=false_clearances,
        false_clearance_scenarios=len(false_clearance_scenario_ids),
        attention_scenarios=len(attention_scenario_ids),
        true_clearances=true_clearances,
        attention_recall=(1.0 - false_clearances / len(attention) if attention else 1.0),
        clearance_coverage=cleared / len(records) if records else 0.0,
        clearance_error_rate=false_clearances / cleared if cleared else 0.0,
        observed_false_clear_rate=(
            len(false_clearance_scenario_ids) / len(attention_scenario_ids)
            if attention_scenario_ids
            else 0.0
        ),
        upper_false_clear_rate=exact_binomial_upper_bound(
            len(false_clearance_scenario_ids),
            len(attention_scenario_ids),
            confidence_level,
        ),
        accuracy=correct / len(records) if records else 0.0,
    )


def _group_metrics(
    records: list[DecisionRecord],
    decisions: list[PolicyDecision],
    keys: list[tuple[str, ...]],
    confidence_level: float,
) -> dict[str, PolicyMetrics]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record_keys in enumerate(keys):
        for key in record_keys:
            grouped[key].append(index)
    return {
        key: _metrics(
            [records[index] for index in indices],
            [decisions[index] for index in indices],
            confidence_level,
        )
        for key, indices in sorted(grouped.items())
    }


def _pair_consistency(
    records: list[DecisionRecord],
    decisions: list[PolicyDecision],
) -> float:
    pairs: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        pairs[record.pair_id].append(index)
    consistent = sum(
        len(indices) == 2
        and all(
            decisions[index].requires_human_lawyer_attention
            == records[index].requires_human_lawyer_attention
            for index in indices
        )
        for indices in pairs.values()
    )
    return consistent / len(pairs) if pairs else 0.0


def _paraphrase_invariance(
    records: list[DecisionRecord],
    decisions: list[PolicyDecision],
) -> float:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record.scenario_id, record.label)].append(index)
    invariant = sum(
        len(indices) == 2
        and len({decisions[index].requires_human_lawyer_attention for index in indices}) == 1
        for indices in groups.values()
    )
    return invariant / len(groups) if groups else 0.0


def _safe_score_monotonicity(
    records: list[DecisionRecord],
    decisions: list[PolicyDecision],
) -> float:
    pairs: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        pairs[record.pair_id].append(index)
    valid = 0
    monotonic = 0
    for indices in pairs.values():
        if len(indices) != 2:
            continue
        safe = next(
            (index for index in indices if not records[index].requires_human_lawyer_attention),
            None,
        )
        risky = next((index for index in indices if index != safe), None)
        if safe is None or risky is None:
            continue
        valid += 1
        monotonic += decisions[safe].clearance_score > decisions[risky].clearance_score
    return monotonic / valid if valid else 0.0


def evaluate_policy_split(
    split: str,
    data_dir: Path = BENCHMARK_DIR,
    model_dir: Path = POLICY_MODEL_DIR,
    output_dir: Path = PROJECT_ROOT / "evaluation" / "policy-v2",
    *,
    device: str = "auto",
    batch_size: int = 32,
) -> PolicyEvaluationReport:
    """Evaluate one benchmark split through the persisted policy ensemble."""
    split_path = data_dir / f"{split}.jsonl"
    records = read_jsonl(split_path)
    model = PolicyDecisionModel(model_dir, device=device)
    decisions = list(
        model.predict_many(
            [record.text for record in records],
            batch_size=batch_size,
        )
    )
    confidence = model.metadata.confidence_level
    ood_indices = [index for index, record in enumerate(records) if not record.in_scope]
    routing_reasons: dict[str, int] = defaultdict(int)
    for decision in decisions:
        routing_reasons[decision.routing_reason] += 1
    split_sha256, manifest_sha256, benchmark_seed = _benchmark_identity(split, split_path)
    report = PolicyEvaluationReport(
        split=split,
        artifact_fingerprint=model.artifact_fingerprint,
        policy_version=model.metadata.policy_version,
        base_model=model.metadata.base_model,
        base_model_revision=model.metadata.base_model_revision,
        benchmark_split_sha256=split_sha256,
        benchmark_manifest_sha256=manifest_sha256,
        benchmark_seed=benchmark_seed,
        metrics=_metrics(records, decisions, confidence),
        counterfactual_pair_consistency=_pair_consistency(records, decisions),
        paraphrase_invariance=_paraphrase_invariance(records, decisions),
        safe_score_monotonicity=_safe_score_monotonicity(records, decisions),
        out_of_distribution_attention_rate=(
            sum(decisions[index].requires_human_lawyer_attention for index in ood_indices)
            / len(ood_indices)
            if ood_indices
            else 0.0
        ),
        family_metrics=_group_metrics(
            records,
            decisions,
            [(record.family,) for record in records],
            confidence,
        ),
        generator_metrics=_group_metrics(
            records,
            decisions,
            [(record.generator,) for record in records],
            confidence,
        ),
        capability_metrics=_group_metrics(
            records,
            decisions,
            [record.capabilities for record in records],
            confidence,
        ),
        routing_reasons=dict(sorted(routing_reasons.items())),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split}.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
