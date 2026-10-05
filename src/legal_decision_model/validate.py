"""Dataset integrity and synthetic-only validation."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from legal_decision_model.constants import BENCHMARK_DIR, LABELS, SPLIT_SIZES
from legal_decision_model.policy import evaluate_facts
from legal_decision_model.policy_schema import PolicySpec, load_policy
from legal_decision_model.records import DecisionRecord, read_jsonl

FORBIDDEN_DATA_PATTERNS = {
    "internal GitHub host": re.compile(r"\bgithub\.(?:net|corp|internal)\b", re.IGNORECASE),
    "Microsoft tenant URL": re.compile(r"\b[a-z0-9-]+\.sharepoint\.com\b", re.IGNORECASE),
    "Slack archive URL": re.compile(r"\bslack\.com/archives/", re.IGNORECASE),
    "local home path": re.compile(r"/Users/|[A-Z]:\\\\Users\\\\"),
    "corporate email": re.compile(r"\b[A-Z0-9._%+-]+@(?:github|microsoft)\.com\b", re.IGNORECASE),
    "credential-like token": re.compile(r"\b(?:ghp_|github_pat_|sk-[A-Za-z0-9])"),
}


@dataclass(frozen=True)
class ValidationReport:
    """Summary of validated records."""

    total: int
    splits: dict[str, int]
    labels: dict[str, int]
    families: dict[str, int]
    generators: dict[str, int]
    scenario_types: dict[str, int]
    capabilities: dict[str, int]
    pairs: int
    scenarios: int


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _validate_record(
    record: DecisionRecord,
    *,
    require_v2_metadata: bool,
    policy: PolicySpec | None = None,
) -> list[str]:
    policy = load_policy() if policy is None else policy
    errors = []
    if not record.synthetic:
        errors.append(f"{record.record_id}: synthetic flag must be true")
    if record.label not in LABELS:
        errors.append(f"{record.record_id}: unknown label {record.label!r}")
    if require_v2_metadata:
        expected_facts = set(policy.fact_names)
        actual_facts = set(record.facts)
        if actual_facts != expected_facts:
            missing = sorted(expected_facts - actual_facts)
            extra = sorted(actual_facts - expected_facts)
            errors.append(
                f"{record.record_id}: facts must exactly match the policy schema "
                f"(missing={missing}, extra={extra})"
            )
    try:
        facts = policy.validate_facts(record.facts)
        decision = evaluate_facts(facts, policy)
    except ValueError as exc:
        errors.append(f"{record.record_id}: {exc}")
        return errors
    if decision.label != record.label:
        errors.append(
            f"{record.record_id}: policy gives {decision.label}, record gives {record.label}"
        )
    if decision.reasons != record.reasons:
        errors.append(f"{record.record_id}: reasons do not match policy")
    if require_v2_metadata:
        if (
            record.in_scope is False
            and "outside_approved_playbook" in facts
            and not facts["outside_approved_playbook"]
        ):
            errors.append(
                f"{record.record_id}: out-of-scope rows must set outside_approved_playbook"
            )
        if not record.scenario_id or not record.rendering_id:
            errors.append(f"{record.record_id}: scenario and rendering IDs are required")
        if len(set(record.capabilities)) != len(record.capabilities):
            errors.append(f"{record.record_id}: capabilities contain duplicates")
        if (
            "conflicting_accounts" in record.capabilities
            and record.requires_human_lawyer_attention
            and "incomplete_or_conflicting_facts" in facts
            and not facts["incomplete_or_conflicting_facts"]
        ):
            errors.append(
                f"{record.record_id}: conflicting accounts require the corresponding policy fact"
            )
    for name, pattern in FORBIDDEN_DATA_PATTERNS.items():
        if pattern.search(record.text):
            errors.append(f"{record.record_id}: text matches forbidden {name}")
    return errors


def validate_dataset(
    data_dir: Path = BENCHMARK_DIR,
    split_sizes: dict[str, int] | None = None,
    *,
    require_v2_metadata: bool = True,
) -> ValidationReport:
    """Validate labels, split isolation, pairs, duplicates, and public-safe content."""
    records: list[DecisionRecord] = []
    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    sizes = SPLIT_SIZES if split_sizes is None else split_sizes
    policy = load_policy()
    for split, expected in sizes.items():
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        split_records = read_jsonl(path)
        split_counts[split] = len(split_records)
        if len(split_records) != expected:
            errors.append(f"{split}: expected {expected} rows, found {len(split_records)}")
        for record in split_records:
            if record.split != split:
                errors.append(f"{record.record_id}: stored in {split}, names {record.split}")
            errors.extend(
                _validate_record(
                    record,
                    require_v2_metadata=require_v2_metadata,
                    policy=policy,
                )
            )
        records.extend(split_records)

    ids = [record.record_id for record in records]
    if len(ids) != len(set(ids)):
        errors.append("record IDs are not unique")
    texts = [_normalized(record.text) for record in records]
    if len(texts) != len(set(texts)):
        errors.append("normalized request texts are not unique")

    templates: dict[str, set[str]] = defaultdict(set)
    for record in records:
        templates[record.template_id].add(record.split)
    leaked = [template for template, splits in templates.items() if len(splits) > 1]
    if leaked:
        errors.append(f"{len(leaked)} templates cross split boundaries")
    if require_v2_metadata:
        generators_by_split = {
            split: {record.generator for record in records if record.split == split}
            for split in sizes
        }
        if generators_by_split.get("train", set()) & generators_by_split.get("test", set()):
            errors.append("train and test use the same generator")
        if generators_by_split.get("train", set()) & generators_by_split.get("challenge", set()):
            errors.append("train and challenge use the same generator")

    pairs: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        pairs[record.pair_id].append(record)
    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            errors.append(f"{pair_id}: expected 2 records, found {len(pair)}")
            continue
        if {record.label for record in pair} != set(LABELS):
            errors.append(f"{pair_id}: pair does not contain both labels")
        differing = tuple(key for key in pair[0].facts if pair[0].facts[key] != pair[1].facts[key])
        if require_v2_metadata:
            expected_changed = pair[0].changed_facts
            if set(differing) != set(expected_changed) or not differing:
                errors.append(
                    f"{pair_id}: changed facts {differing} do not match metadata {expected_changed}"
                )
            if pair[0].scenario_id != pair[1].scenario_id:
                errors.append(f"{pair_id}: records do not share a scenario")
        elif len(differing) != 1:
            errors.append(f"{pair_id}: expected one changed fact, found {differing}")

    scenarios: dict[str, list[DecisionRecord]] = defaultdict(list)
    if require_v2_metadata:
        for record in records:
            scenarios[record.scenario_id].append(record)
        for scenario_id, scenario in scenarios.items():
            if len(scenario) != 4:
                errors.append(f"{scenario_id}: expected four records, found {len(scenario)}")
                continue
            if len({record.rendering_id for record in scenario}) != 2:
                errors.append(f"{scenario_id}: expected two renderings")
            if Counter(record.label for record in scenario) != Counter(
                {LABELS[0]: 2, LABELS[1]: 2}
            ):
                errors.append(f"{scenario_id}: expected two records per label")
            grouped_facts: dict[str, set[tuple[tuple[str, bool], ...]]] = defaultdict(set)
            for record in scenario:
                grouped_facts[record.label].add(tuple(record.facts.items()))
            if any(len(values) != 1 for values in grouped_facts.values()):
                errors.append(f"{scenario_id}: paraphrases do not preserve structured facts")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:25])
        suffix = "" if len(errors) <= 25 else f"\n- ... and {len(errors) - 25} more"
        raise ValueError(f"dataset validation failed:\n{preview}{suffix}")

    return ValidationReport(
        total=len(records),
        splits=dict(split_counts),
        labels=dict(Counter(record.label for record in records)),
        families=dict(Counter(record.family for record in records)),
        generators=dict(Counter(record.generator for record in records)),
        scenario_types=dict(Counter(record.scenario_type for record in records)),
        capabilities=dict(
            Counter(capability for record in records for capability in record.capabilities)
        ),
        pairs=len(pairs),
        scenarios=len(scenarios),
    )
