"""Dataset integrity and synthetic-only validation."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from legal_decision_model.constants import DATA_DIR, LABELS, SPLIT_SIZES
from legal_decision_model.policy import ScenarioFacts, evaluate_policy
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
    pairs: int


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _validate_record(record: DecisionRecord) -> list[str]:
    errors = []
    if not record.synthetic:
        errors.append(f"{record.record_id}: synthetic flag must be true")
    if record.label not in LABELS:
        errors.append(f"{record.record_id}: unknown label {record.label!r}")
    try:
        decision = evaluate_policy(ScenarioFacts.from_dict(record.facts))
    except ValueError as exc:
        errors.append(f"{record.record_id}: {exc}")
        return errors
    if decision.label != record.label:
        errors.append(
            f"{record.record_id}: policy gives {decision.label}, record gives {record.label}"
        )
    if decision.reasons != record.reasons:
        errors.append(f"{record.record_id}: reasons do not match policy")
    for name, pattern in FORBIDDEN_DATA_PATTERNS.items():
        if pattern.search(record.text):
            errors.append(f"{record.record_id}: text matches forbidden {name}")
    return errors


def validate_dataset(data_dir: Path = DATA_DIR) -> ValidationReport:
    """Validate labels, split isolation, pairs, duplicates, and public-safe content."""
    records: list[DecisionRecord] = []
    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    for split, expected in SPLIT_SIZES.items():
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
            errors.extend(_validate_record(record))
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

    pairs: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        pairs[record.pair_id].append(record)
    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            errors.append(f"{pair_id}: expected 2 records, found {len(pair)}")
            continue
        if {record.label for record in pair} != set(LABELS):
            errors.append(f"{pair_id}: pair does not contain both labels")
        differing = [key for key in pair[0].facts if pair[0].facts[key] != pair[1].facts[key]]
        if len(differing) != 1:
            errors.append(f"{pair_id}: expected one changed policy fact, found {differing}")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:25])
        suffix = "" if len(errors) <= 25 else f"\n- ... and {len(errors) - 25} more"
        raise ValueError(f"dataset validation failed:\n{preview}{suffix}")

    return ValidationReport(
        total=len(records),
        splits=dict(split_counts),
        labels=dict(Counter(record.label for record in records)),
        families=dict(Counter(record.family for record in records)),
        pairs=len(pairs),
    )
