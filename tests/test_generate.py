from pathlib import Path

from legal_decision_model.constants import LABELS, SPLIT_SIZES
from legal_decision_model.generate import generate_dataset
from legal_decision_model.records import read_jsonl
from legal_decision_model.validate import validate_dataset


def test_generation_is_reproducible_and_valid(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(first, seed=42)
    generate_dataset(second, seed=42)

    for split, size in SPLIT_SIZES.items():
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
        records = read_jsonl(first / f"{split}.jsonl")
        assert len(records) == size
        assert {record.label for record in records} == set(LABELS)

    report = validate_dataset(first)
    assert report.total == sum(SPLIT_SIZES.values())
    assert report.pairs == report.total // 2


def test_each_pair_changes_one_policy_fact(tmp_path: Path) -> None:
    generate_dataset(tmp_path, seed=7)
    records = read_jsonl(tmp_path / "challenge.jsonl")
    pairs: dict[str, list[dict[str, bool]]] = {}
    for record in records:
        pairs.setdefault(record.pair_id, []).append(record.facts)
    for pair in pairs.values():
        changed = [key for key in pair[0] if pair[0][key] != pair[1][key]]
        assert len(changed) == 1
