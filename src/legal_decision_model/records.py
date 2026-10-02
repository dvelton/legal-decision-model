"""Dataset record types and JSONL helpers."""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legal_decision_model.policy import ScenarioFacts


@dataclass(frozen=True)
class DecisionRecord:
    """One wholly synthetic request and its deterministic policy label."""

    record_id: str
    pair_id: str
    split: str
    family: str
    style: str
    template_id: str
    text: str
    facts: dict[str, bool]
    label: str
    reasons: tuple[str, ...]
    synthetic: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DecisionRecord":
        """Parse a serialized record."""
        values = dict(raw)
        values["reasons"] = tuple(values["reasons"])
        values["facts"] = ScenarioFacts.from_dict(values["facts"]).to_dict()
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        return asdict(self)


def read_jsonl(path: Path) -> list[DecisionRecord]:
    """Read decision records from a JSONL file."""
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        records.append(DecisionRecord.from_dict(raw))
    return records


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield raw JSON objects from a JSONL file."""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        yield value


def write_jsonl(path: Path, records: list[DecisionRecord]) -> None:
    """Write records as stable UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True) + "\n" for record in records
    )
    path.write_text(content, encoding="utf-8")
