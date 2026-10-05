from legal_decision_model.constants import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    LABELS,
    LEGACY_POLICY_VERSION,
    SITE_NAME,
)
from legal_decision_model.model_artifacts import ModelMetadata


def test_public_v1_model_metadata_remains_loadable() -> None:
    metadata = ModelMetadata.from_dict(
        {
            "site": SITE_NAME,
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_MODEL_REVISION,
            "labels": list(LABELS),
            "field": SITE_NAME,
            "temperature": 1.0,
            "clearance_threshold": 0.9,
            "max_false_clear_rate": 0.0,
            "policy_version": LEGACY_POLICY_VERSION,
            "trained_at": "2026-10-02T00:00:00+00:00",
            "training_rows": 7500,
            "validation_rows": 1000,
            "epochs": 24,
            "seed": 20261002,
            "device": "mps",
            "max_len": 512,
            "head_max_len": 192,
            "spaced_labels": True,
            "validation_metrics": {"accuracy": 1.0},
        }
    )
    assert metadata.policy_version == LEGACY_POLICY_VERSION
