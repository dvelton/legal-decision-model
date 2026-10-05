"""Dataset record types and JSONL helpers."""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legal_decision_model.constants import REQUIRES_ATTENTION


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
    generator: str = "procedural-a"
    scenario_type: str = "counterfactual"
    scenario_id: str = ""
    rendering_id: str = ""
    capabilities: tuple[str, ...] = ()
    changed_facts: tuple[str, ...] = ()
    in_scope: bool = True
    synthetic: bool = True

    @property
    def requires_human_lawyer_attention(self) -> bool:
        """Whether the deterministic label requires human attention."""
        return self.label == REQUIRES_ATTENTION

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DecisionRecord":
        """Parse a serialized record."""
        values = dict(raw)
        values["reasons"] = tuple(values["reasons"])
        values["capabilities"] = tuple(values.get("capabilities", ()))
        values["changed_facts"] = tuple(values.get("changed_facts", ()))
        facts = values["facts"]
        if not isinstance(facts, dict):
            raise ValueError("facts must be an object")
        if any(not isinstance(name, str) for name in facts):
            raise ValueError("policy fact names must be strings")
        if any(not isinstance(value, bool) for value in facts.values()):
            raise ValueError("every policy fact must be a boolean")
        values["facts"] = dict(facts)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        return asdict(self)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the exact public v0.1 record shape."""
        return {
            "record_id": self.record_id,
            "pair_id": self.pair_id,
            "split": self.split,
            "family": self.family,
            "style": self.style,
            "template_id": self.template_id,
            "text": self.text,
            "facts": self.facts,
            "label": self.label,
            "reasons": self.reasons,
            "synthetic": self.synthetic,
        }


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


def write_jsonl(
    path: Path,
    records: list[DecisionRecord],
    *,
    legacy: bool = False,
) -> None:
    """Write records as stable UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(
            record.to_legacy_dict() if legacy else record.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")
