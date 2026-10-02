"""Train a small task head on the frozen Laya encoder."""

import json
import math
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from stuntd.serve.decider import Decider
from stuntd.train.artifacts import SiteModel
from stuntd.train.dataset import Item, SiteDataset
from stuntd.train.layout import Layout
from stuntd.train.metrics import fit_temperature
from stuntd.train.trainer import LayaTrainer

from legal_decision_model.calibration import (
    ScoredExample,
    evaluate_scores,
    select_clearance_threshold,
)
from legal_decision_model.constants import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    DATA_DIR,
    LABELS,
    MODEL_DIR,
    POLICY_VERSION,
    SITE_NAME,
)
from legal_decision_model.model_artifacts import (
    HEAD_FILE,
    MODEL_METADATA_FILE,
    ModelMetadata,
    load_metadata,
    save_metadata,
)
from legal_decision_model.records import DecisionRecord, read_jsonl
from legal_decision_model.revisions import pinned_laya_revision
from legal_decision_model.validate import validate_dataset


@dataclass(frozen=True)
class TrainSummary:
    """Result of a completed training run."""

    model_dir: Path
    metadata: ModelMetadata


def _items(records: list[DecisionRecord], offset: float) -> tuple[Item, ...]:
    label_index = {label: index for index, label in enumerate(LABELS)}
    return tuple(
        Item(record.text, label_index[record.label], offset + index)
        for index, record in enumerate(records)
    )


def _site_model(
    temperature: float,
    layout: Layout,
    training_rows: int,
    validation_rows: int,
) -> SiteModel:
    return SiteModel(
        site=SITE_NAME,
        kind="choice",
        field=SITE_NAME,
        labels=list(LABELS),
        base_model=BASE_MODEL,
        temperature=temperature,
        threshold=None,
        target_agreement=1.0,
        trained_at=0.0,
        n_train=training_rows,
        n_holdout=validation_rows,
        agreement=0.0,
        coverage=0.0,
        covered_agreement=0.0,
        ece=0.0,
        per_class={},
        confident_errors=[],
        curve=[],
        max_len=layout.max_len,
        head_max_len=layout.head_max_len,
        spaced_labels=layout.spaced_labels,
    )


def _calibrate_saved_head(
    head_path: Path,
    validation_records: list[DecisionRecord],
    layout: Layout,
    *,
    device: str,
    training_rows: int,
    max_false_clear_rate: float,
) -> tuple[float, float | None, dict[str, int | float]]:
    with pinned_laya_revision(BASE_MODEL_REVISION):
        decider = Decider(BASE_MODEL, device=device, lazy=False)
    uncalibrated = _site_model(1.0, layout, training_rows, len(validation_records))
    raw_probabilities = [
        decider.decide(uncalibrated, head_path, record.text).probabilities
        for record in validation_records
    ]
    raw_logits = [
        [math.log(max(probability, 1e-300)) for probability in probabilities]
        for probabilities in raw_probabilities
    ]
    gold_indices = [LABELS.index(record.label) for record in validation_records]
    temperature = fit_temperature(raw_logits, gold_indices)
    calibrated = _site_model(temperature, layout, training_rows, len(validation_records))
    scored = [
        ScoredExample(
            record.label,
            decider.decide(calibrated, head_path, record.text).probabilities[0],
        )
        for record in validation_records
    ]
    threshold = select_clearance_threshold(scored, max_false_clear_rate)
    metrics = evaluate_scores(scored, threshold)
    if metrics.false_clear_rate > max_false_clear_rate + 1e-12:
        raise RuntimeError("serialized model does not meet the configured false-clear limit")
    return temperature, threshold, metrics.to_dict()


def train_model(
    data_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
    *,
    device: str = "mps",
    epochs: int = 24,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    seed: int = 20261002,
    max_false_clear_rate: float = 0.0,
) -> TrainSummary:
    """Train, calibrate, and save the conservative legal-attention head."""
    validate_dataset(data_dir)
    train_records = read_jsonl(data_dir / "train.jsonl")
    validation_records = read_jsonl(data_dir / "validation.jsonl")
    dataset = SiteDataset(
        site=SITE_NAME,
        kind="choice",
        field=SITE_NAME,
        labels=LABELS,
        train=_items(train_records, 0.0),
        holdout=_items(validation_records, float(len(train_records) + 1)),
        deduplicated=0,
        dropped=0,
    )

    working = model_dir.with_name(model_dir.name + ".work")
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(parents=True)
    try:
        with pinned_laya_revision(BASE_MODEL_REVISION):
            trainer = LayaTrainer(
                BASE_MODEL,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                cache_encoder=True,
            )
        trained = trainer(dataset, working / HEAD_FILE)
        del trainer
        temperature, threshold, validation_metrics = _calibrate_saved_head(
            working / HEAD_FILE,
            validation_records,
            trained.layout,
            device=device,
            training_rows=len(train_records),
            max_false_clear_rate=max_false_clear_rate,
        )
        metadata = ModelMetadata(
            site=SITE_NAME,
            base_model=BASE_MODEL,
            base_model_revision=BASE_MODEL_REVISION,
            labels=LABELS,
            field=SITE_NAME,
            temperature=temperature,
            clearance_threshold=threshold,
            max_false_clear_rate=max_false_clear_rate,
            policy_version=POLICY_VERSION,
            trained_at=datetime.now(UTC).isoformat(),
            training_rows=len(train_records),
            validation_rows=len(validation_records),
            epochs=epochs,
            seed=seed,
            device=device,
            max_len=trained.layout.max_len,
            head_max_len=trained.layout.head_max_len,
            spaced_labels=trained.layout.spaced_labels,
            validation_metrics=validation_metrics,
        )
        save_metadata(working / MODEL_METADATA_FILE, metadata)
        (working / "training-summary.json").write_text(
            json.dumps(
                {
                    "architecture": "frozen Laya encoder with a trained stuntd decision head",
                    "metadata": metadata.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if model_dir.exists():
            shutil.rmtree(model_dir)
        working.replace(model_dir)
    except BaseException:
        if working.exists():
            shutil.rmtree(working)
        raise
    return TrainSummary(model_dir, metadata)


def recalibrate_model(
    data_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
    *,
    device: str = "mps",
) -> TrainSummary:
    """Recalibrate an existing serialized head through the serving path."""
    metadata = load_metadata(model_dir)
    validation_records = read_jsonl(data_dir / "validation.jsonl")
    layout = Layout(metadata.max_len, metadata.head_max_len, metadata.spaced_labels)
    temperature, threshold, validation_metrics = _calibrate_saved_head(
        model_dir / HEAD_FILE,
        validation_records,
        layout,
        device=device,
        training_rows=metadata.training_rows,
        max_false_clear_rate=metadata.max_false_clear_rate,
    )
    updated = replace(
        metadata,
        temperature=temperature,
        clearance_threshold=threshold,
        validation_metrics=validation_metrics,
    )
    save_metadata(model_dir / MODEL_METADATA_FILE, updated)
    (model_dir / "training-summary.json").write_text(
        json.dumps(
            {
                "architecture": "frozen Laya encoder with a trained stuntd decision head",
                "metadata": updated.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainSummary(model_dir, updated)
