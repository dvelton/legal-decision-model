"""Persisted model metadata."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from legal_decision_model.constants import (
    LABELS,
    LEGACY_POLICY_VERSION,
    POLICY_VERSION,
    SITE_NAME,
)

HEAD_FILE = "head.safetensors"
MODEL_METADATA_FILE = "model.json"
SUPPORTED_POLICY_VERSIONS = frozenset({LEGACY_POLICY_VERSION, POLICY_VERSION})


@dataclass(frozen=True)
class ModelMetadata:
    """Everything needed to load and interpret a trained decision head."""

    site: str
    base_model: str
    base_model_revision: str
    labels: tuple[str, str]
    field: str
    temperature: float
    clearance_threshold: float | None
    max_false_clear_rate: float
    policy_version: str
    trained_at: str
    training_rows: int
    validation_rows: int
    epochs: int
    seed: int
    device: str
    max_len: int
    head_max_len: int
    spaced_labels: bool
    validation_metrics: dict[str, int | float]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelMetadata":
        """Parse and validate metadata."""
        values = dict(raw)
        values["labels"] = tuple(values["labels"])
        metadata = cls(**values)
        if metadata.site != SITE_NAME:
            raise ValueError(f"unexpected site {metadata.site!r}")
        if metadata.labels != LABELS:
            raise ValueError(f"unexpected label order {metadata.labels!r}")
        if metadata.policy_version not in SUPPORTED_POLICY_VERSIONS:
            raise ValueError(f"unexpected policy version {metadata.policy_version!r}")
        if metadata.clearance_threshold is not None and not (
            0 <= metadata.clearance_threshold <= 1
        ):
            raise ValueError("clearance threshold must be between 0 and 1 or null")
        return metadata

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata."""
        return asdict(self)


def save_metadata(path: Path, metadata: ModelMetadata) -> None:
    """Write stable model metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_metadata(model_dir: Path) -> ModelMetadata:
    """Read model metadata from a trained artifact directory."""
    path = model_dir / MODEL_METADATA_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"trained model metadata not found at {path}; run `legal-decision-model train`"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object")
    return ModelMetadata.from_dict(raw)
