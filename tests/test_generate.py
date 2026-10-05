import json
from dataclasses import replace
from pathlib import Path

import legal_decision_model.generate as legacy_generate
from legal_decision_model.benchmark import generate_benchmark, generate_benchmark_split
from legal_decision_model.constants import LABELS
from legal_decision_model.policy_schema import load_policy
from legal_decision_model.records import read_jsonl
from legal_decision_model.validate import validate_dataset


def test_generation_is_reproducible_and_valid(tmp_path: Path) -> None:
    sizes = {"train": 40, "validation": 20, "test": 20, "challenge": 20}
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_benchmark(first, seed=42, split_sizes=sizes)
    generate_benchmark(second, seed=42, split_sizes=sizes)

    for split, size in sizes.items():
        assert (first / f"{split}.jsonl").read_bytes() == (second / f"{split}.jsonl").read_bytes()
        records = read_jsonl(first / f"{split}.jsonl")
        assert len(records) == size
        assert {record.label for record in records} == set(LABELS)

    report = validate_dataset(first, split_sizes=sizes)
    assert report.total == sum(sizes.values())
    assert report.pairs == report.total // 2
    assert report.scenarios == report.total // 4
    assert report.generators["procedural-b-independent"] == sizes["test"]


def test_each_pair_changes_one_policy_fact(tmp_path: Path) -> None:
    sizes = {"train": 40, "validation": 20, "test": 20, "challenge": 20}
    generate_benchmark(tmp_path, seed=7, split_sizes=sizes)
    records = read_jsonl(tmp_path / "challenge.jsonl")
    pairs = {}
    for record in records:
        pairs.setdefault(record.pair_id, []).append(record)
    for pair in pairs.values():
        changed = {key for key in pair[0].facts if pair[0].facts[key] != pair[1].facts[key]}
        assert changed == set(pair[0].changed_facts)


def test_behavioral_text_matches_structured_facts(tmp_path: Path) -> None:
    sizes = {"train": 80, "validation": 80, "test": 80, "challenge": 80}
    generate_benchmark(tmp_path, seed=11, split_sizes=sizes)
    records = [record for split in sizes for record in read_jsonl(tmp_path / f"{split}.jsonl")]
    conflicting = [
        record
        for record in records
        if "conflicting_accounts" in record.capabilities and record.requires_human_lawyer_attention
    ]
    assert conflicting
    assert all(record.facts["incomplete_or_conflicting_facts"] for record in conflicting)
    assert all("incomplete_or_conflicting_facts" in record.changed_facts for record in conflicting)
    long_records = [record for record in records if "long_context" in record.capabilities]
    assert long_records
    assert min(len(record.text) for record in long_records) > 4_000


def test_v2_generator_uses_compiled_policy_fact_schema() -> None:
    policy = load_policy()
    removed = "employment_action"
    customized = replace(
        policy,
        version="custom-1.0",
        triggers=tuple(name for name in policy.triggers if name != removed),
        trigger_reasons={
            name: reason for name, reason in policy.trigger_reasons.items() if name != removed
        },
    )
    records = generate_benchmark_split(
        "validation",
        80,
        17,
        customized,
    )
    assert records
    assert all(set(record.facts) == set(customized.fact_names) for record in records)
    assert all(removed not in record.facts for record in records)


def test_legacy_generator_preserves_public_record_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(legacy_generate, "LEGACY_SPLIT_SIZES", {"train": 2})
    legacy_generate.generate_dataset(tmp_path, seed=20261002)
    raw = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(raw) == {
        "record_id",
        "pair_id",
        "split",
        "family",
        "style",
        "template_id",
        "text",
        "facts",
        "label",
        "reasons",
        "synthetic",
    }
