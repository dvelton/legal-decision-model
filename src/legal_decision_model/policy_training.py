"""Train the multi-task Laya policy-head ensemble."""

import hashlib
import json
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from legal_decision_model.constants import (
    ARTIFACTS_DIR,
    BASE_MODEL,
    BASE_MODEL_REVISION,
    BENCHMARK_DIR,
    POLICY_MODEL_DIR,
    POLICY_TRAINING_SEED,
)
from legal_decision_model.generate import FAMILIES
from legal_decision_model.policy_model import (
    FINAL_TASK,
    FeatureCache,
    LayaFeatureEncoder,
    PolicyHead,
    PolicyHeadConfig,
    PolicyModelMetadata,
    cosine_distance,
    load_head,
    quantile,
    safe_score,
    save_head,
    save_policy_metadata,
    split_text_windows,
    task_names,
    task_targets,
)
from legal_decision_model.policy_schema import PolicySpec, load_policy, policy_fingerprint
from legal_decision_model.records import DecisionRecord, read_jsonl
from legal_decision_model.risk_control import (
    RiskExample,
    RiskSelection,
    exact_binomial_upper_bound,
    select_risk_controlled_threshold,
)
from legal_decision_model.validate import validate_dataset

METADATA_FILE = "metadata.json"
RISK_REPORT_FILE = "risk-control.json"


@dataclass(frozen=True)
class PolicyTrainingSummary:
    """Completed policy-model training run."""

    model_dir: Path
    metadata: PolicyModelMetadata


@dataclass(frozen=True)
class DocumentFeatureCache:
    """Flattened serving windows and their record boundaries."""

    features: torch.Tensor
    window_counts: tuple[int, ...]


def _classification_weights(targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    return (negatives / positives.clamp(min=1)).clamp(min=0.25, max=8.0)


def _pair_indices(
    records: tuple[DecisionRecord, ...],
    safe_label: str,
) -> list[tuple[int, int, tuple[str, ...]]]:
    pairs: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        pairs[record.pair_id].append(index)
    output = []
    for pair in pairs.values():
        if len(pair) != 2:
            continue
        safe = next(index for index in pair if records[index].label == safe_label)
        risky = next(index for index in pair if index != safe)
        output.append((safe, risky, records[risky].changed_facts))
    return output


def _consistency_indices(records: tuple[DecisionRecord, ...]) -> list[tuple[int, int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record.scenario_id, record.label)].append(index)
    return [(indices[0], indices[1]) for indices in groups.values() if len(indices) == 2]


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _normalize(
    features: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (features.float() - mean) / scale


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _train_member(
    cache: FeatureCache,
    config: PolicyHeadConfig,
    policy: PolicySpec,
    *,
    member_index: int,
    seed: int,
    device: torch.device,
) -> PolicyHead:
    torch.manual_seed(seed)
    random.seed(seed)
    head = PolicyHead(config).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=policy.model.learning_rate)
    task_targets = cache.task_targets
    family_targets = cache.family_targets
    task_weight = _classification_weights(task_targets).to(device)
    pair_indices = _pair_indices(cache.records, policy.no_attention_label)
    consistency_indices = _consistency_indices(cache.records)
    task_index = {name: index for index, name in enumerate(config.task_names)}
    final_index = task_index[FINAL_TASK]
    row_indices = list(range(len(cache.records)))
    batch_size = policy.model.batch_size

    for epoch in range(policy.model.epochs):
        print(
            f"Training ensemble member {member_index + 1}/{policy.model.ensemble_size}, "
            f"epoch {epoch + 1}/{policy.model.epochs}",
            flush=True,
        )
        head.train()
        random.Random(seed + epoch).shuffle(row_indices)
        for batch_indices in _chunks(row_indices, batch_size):
            index = torch.tensor(batch_indices)
            features = cache.features[index].to(device, torch.float32)
            targets = task_targets[index].to(device)
            families = family_targets[index].to(device)
            task_logits, family_logits = head(features)
            task_loss = nn.functional.binary_cross_entropy_with_logits(
                task_logits,
                targets,
                pos_weight=task_weight,
            )
            family_loss = nn.functional.cross_entropy(family_logits, families)
            loss = task_loss + 0.15 * family_loss
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        random.Random(seed + epoch + 10_000).shuffle(pair_indices)
        for pair_batch in [
            pair_indices[start : start + batch_size]
            for start in range(0, len(pair_indices), batch_size)
        ]:
            safe_indices = torch.tensor([item[0] for item in pair_batch])
            risky_indices = torch.tensor([item[1] for item in pair_batch])
            safe_logits, _ = head(cache.features[safe_indices].to(device, torch.float32))
            risky_logits, _ = head(cache.features[risky_indices].to(device, torch.float32))
            final_margin = nn.functional.relu(
                policy.model.ranking_margin
                - (risky_logits[:, final_index] - safe_logits[:, final_index])
            ).mean()
            trigger_losses = []
            for row, (_, _, changed) in enumerate(pair_batch):
                for name in changed:
                    position = task_index[name]
                    trigger_losses.append(
                        nn.functional.relu(
                            torch.tensor(policy.model.ranking_margin, device=device)
                            - (risky_logits[row, position] - safe_logits[row, position])
                        )
                    )
            trigger_margin = (
                torch.stack(trigger_losses).mean()
                if trigger_losses
                else torch.tensor(0.0, device=device)
            )
            loss = policy.model.ranking_weight * (final_margin + trigger_margin)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        random.Random(seed + epoch + 20_000).shuffle(consistency_indices)
        for consistency_batch in [
            consistency_indices[start : start + batch_size]
            for start in range(0, len(consistency_indices), batch_size)
        ]:
            first = torch.tensor([item[0] for item in consistency_batch])
            second = torch.tensor([item[1] for item in consistency_batch])
            first_logits, _ = head(cache.features[first].to(device, torch.float32))
            second_logits, _ = head(cache.features[second].to(device, torch.float32))
            consistency = nn.functional.mse_loss(
                torch.sigmoid(first_logits),
                torch.sigmoid(second_logits),
            )
            loss = policy.model.consistency_weight * consistency
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    head.eval()
    return head


def _cached_feature_cache(
    records: list[DecisionRecord],
    policy: PolicySpec,
    encoder: LayaFeatureEncoder,
    families: tuple[str, ...],
    source_path: Path,
    cache_dir: Path,
    batch_size: int,
) -> FeatureCache:
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        digest.update(hashlib.file_digest(handle, "sha256").digest())
    digest.update(BASE_MODEL_REVISION.encode())
    digest.update(b"training-serving-window-v2")
    digest.update(str(encoder.max_tokens).encode())
    digest.update(str(encoder.output_size).encode())
    cache_path = cache_dir / f"{source_path.stem}-{digest.hexdigest()[:20]}.safetensors"
    if cache_path.is_file():
        print(f"Loading cached Laya features from {cache_path}", flush=True)
        features = load_file(str(cache_path))["features"]
        if features.shape != (len(records), encoder.output_size):
            raise ValueError(f"cached feature shape is invalid: {features.shape}")
    else:
        print(f"Encoding {len(records):,} rows from {source_path}", flush=True)
        training_texts = []
        for record in records:
            windows = split_text_windows(
                encoder,
                record.text,
                max_tokens=policy.model.max_tokens,
                window_overlap=policy.model.window_overlap,
                maximum_windows=policy.model.maximum_windows,
            )
            if windows is None:
                raise ValueError(
                    f"{record.record_id}: training record exceeds the maximum window count"
                )
            training_texts.append(windows[-1])
        features, chunk_dir = _resumable_encode(
            training_texts,
            encoder,
            cache_path,
            batch_size=batch_size,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_completed_feature_cache(
            cache_path,
            {"features": features.contiguous()},
            chunk_dir,
        )
        print(f"Saved Laya features to {cache_path}", flush=True)
    family_index = {name: index for index, name in enumerate(families)}
    return FeatureCache(
        features=features,
        task_targets=torch.tensor(
            [task_targets(record, policy) for record in records],
            dtype=torch.float32,
        ),
        family_targets=torch.tensor(
            [family_index[record.family] for record in records],
            dtype=torch.long,
        ),
        records=tuple(records),
    )


def _cached_document_features(
    records: list[DecisionRecord],
    policy: PolicySpec,
    encoder: LayaFeatureEncoder,
    source_path: Path,
    cache_dir: Path,
    batch_size: int,
) -> DocumentFeatureCache:
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        digest.update(hashlib.file_digest(handle, "sha256").digest())
    digest.update(BASE_MODEL_REVISION.encode())
    digest.update(b"serving-windows-v1")
    digest.update(json.dumps(asdict(policy.model), sort_keys=True).encode())
    cache_path = cache_dir / f"{source_path.stem}-windows-{digest.hexdigest()[:20]}.safetensors"
    if cache_path.is_file():
        print(f"Loading cached serving-window features from {cache_path}", flush=True)
        state = load_file(str(cache_path))
        features = state["features"]
        window_counts = tuple(int(value) for value in state["window_counts"].tolist())
        if len(window_counts) != len(records) or sum(window_counts) != len(features):
            raise ValueError("cached serving-window feature boundaries are invalid")
    else:
        windows: list[str] = []
        counts = []
        for record in records:
            record_windows = split_text_windows(
                encoder,
                record.text,
                max_tokens=policy.model.max_tokens,
                window_overlap=policy.model.window_overlap,
                maximum_windows=policy.model.maximum_windows,
            )
            if record_windows is None:
                raise ValueError(
                    f"{record.record_id}: benchmark record exceeds the maximum window count"
                )
            windows.extend(record_windows)
            counts.append(len(record_windows))
        print(
            f"Encoding {len(windows):,} serving windows from {source_path}",
            flush=True,
        )
        features, chunk_dir = _resumable_encode(
            windows,
            encoder,
            cache_path,
            batch_size=batch_size,
        )
        window_counts = tuple(counts)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_completed_feature_cache(
            cache_path,
            {
                "features": features.contiguous(),
                "window_counts": torch.tensor(window_counts, dtype=torch.int32),
            },
            chunk_dir,
        )
        print(f"Saved serving-window features to {cache_path}", flush=True)
    return DocumentFeatureCache(features, window_counts)


def _resumable_encode(
    texts: list[str],
    encoder: LayaFeatureEncoder,
    cache_path: Path,
    *,
    batch_size: int,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, Path]:
    """Encode and persist bounded chunks so interrupted MPS work can resume."""
    rows_per_chunk = chunk_size or batch_size * 8
    if rows_per_chunk < 1:
        raise ValueError("feature-cache chunk size must be positive")
    chunk_dir = cache_path.with_suffix(".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    total = len(texts)
    for start in range(0, total, rows_per_chunk):
        end = min(total, start + rows_per_chunk)
        chunk_path = chunk_dir / f"{start:08d}-{end:08d}.safetensors"
        if chunk_path.is_file():
            chunk = load_file(str(chunk_path))["features"]
            source = "resumed"
        else:
            chunk = encoder.encode(texts[start:end], batch_size=batch_size)
            if not torch.isfinite(chunk).all():
                raise ValueError(f"encoded features are non-finite for rows {start}:{end}")
            temporary = chunk_path.with_name(chunk_path.name + ".tmp")
            save_file({"features": chunk.contiguous()}, str(temporary))
            temporary.replace(chunk_path)
            source = "encoded"
        expected_shape = (end - start, encoder.output_size)
        if chunk.shape != expected_shape:
            raise ValueError(
                f"feature-cache chunk {chunk_path} has shape {chunk.shape}, "
                f"expected {expected_shape}"
            )
        chunks.append(chunk)
        print(
            f"Feature encoding: {end:,}/{total:,} rows ({source})",
            flush=True,
        )
        if encoder.device.type == "mps":
            torch.mps.empty_cache()
    if not chunks:
        return torch.empty((0, encoder.output_size), dtype=torch.float16), chunk_dir
    return torch.cat(chunks), chunk_dir


def _save_completed_feature_cache(
    cache_path: Path,
    state: dict[str, torch.Tensor],
    chunk_dir: Path,
) -> None:
    """Atomically publish a complete cache before removing resumable chunks."""
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    save_file(state, str(temporary))
    temporary.replace(cache_path)
    shutil.rmtree(chunk_dir)


def _ensemble_outputs(
    heads: list[PolicyHead],
    features: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    task_outputs: list[torch.Tensor] = []
    family_outputs: list[torch.Tensor] = []
    disagreement_outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device, torch.float32)
            member_tasks = []
            member_families = []
            for head in heads:
                task_logits, family_logits = head(batch)
                member_tasks.append(torch.sigmoid(task_logits).cpu())
                member_families.append(family_logits.cpu())
            task_outputs.append(torch.stack(member_tasks).mean(dim=0))
            disagreement_outputs.append(torch.stack(member_tasks).std(dim=0, correction=0))
            family_outputs.append(torch.stack(member_families).mean(dim=0))
    return (
        torch.cat(task_outputs),
        torch.cat(family_outputs),
        torch.cat(disagreement_outputs),
    )


def _document_outputs(
    heads: list[PolicyHead],
    features: torch.Tensor,
    window_counts: tuple[int, ...],
    centroids: torch.Tensor,
    policy: PolicySpec,
    task_order: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    window_distances = cosine_distance(features, centroids)
    prerequisite_set = set(policy.prerequisites)
    task_rows = []
    family_rows = []
    disagreement_rows = []
    distance_rows = []
    start = 0
    with torch.no_grad():
        for count in window_counts:
            end = start + count
            batch = features[start:end].to(device, torch.float32)
            member_tasks = []
            member_families = []
            for head in heads:
                task_logits, family_logits = head(batch)
                member_tasks.append(torch.sigmoid(task_logits).cpu())
                member_families.append(family_logits.cpu())
            stacked_tasks = torch.stack(member_tasks)
            task_values = stacked_tasks.mean(dim=0)
            task_rows.append(
                torch.stack(
                    [
                        task_values[:, position].min()
                        if name in prerequisite_set or name == "in_scope"
                        else task_values[:, position].max()
                        for position, name in enumerate(task_order)
                    ]
                )
            )
            family_rows.append(torch.stack(member_families).mean(dim=(0, 1)))
            disagreement_rows.append(stacked_tasks.std(dim=0, correction=0).max(dim=0).values)
            distance_rows.append(window_distances[start:end].max())
            start = end
    return (
        torch.stack(task_rows),
        torch.stack(family_rows),
        torch.stack(disagreement_rows),
        torch.stack(distance_rows),
    )


def _safe_scores(
    task_probabilities: torch.Tensor,
    task_disagreement: torch.Tensor,
    records: tuple[DecisionRecord, ...],
    policy: PolicySpec,
    task_order: tuple[str, ...],
    ood_distance: torch.Tensor,
    ood_threshold: float,
    disagreement_threshold: float,
) -> list[RiskExample]:
    rows = []
    for index, record in enumerate(records):
        probabilities = {
            name: float(task_probabilities[index, position])
            for position, name in enumerate(task_order)
        }
        score = safe_score(probabilities, policy)
        if (
            float(ood_distance[index]) > ood_threshold
            or float(task_disagreement[index].max()) > disagreement_threshold
        ):
            score = 0.0
        rows.append(RiskExample(record.label, score, record.scenario_id))
    return rows


def _selection_dict(selection: RiskSelection) -> dict[str, object]:
    return {
        "threshold": selection.threshold,
        "confidence_level": selection.confidence_level,
        "maximum_false_clear_rate": selection.maximum_false_clear_rate,
        "maximum_false_clearances": selection.maximum_false_clearances,
        "selected": selection.point.to_dict(),
        "curve": [point.to_dict() for point in selection.curve],
    }


def _serving_validation_summary(
    scored: list[RiskExample],
    records: list[DecisionRecord],
    predicted_families: torch.Tensor,
    families: tuple[str, ...],
    global_threshold: float | None,
    family_thresholds: dict[str, float | None],
    policy: PolicySpec,
) -> dict[str, float | int | None]:
    false_clearances = 0
    true_clearances = 0
    attention_examples = sum(row.label == policy.requires_attention_label for row in scored)
    attention_trial_ids = {
        record.scenario_id
        for record, row in zip(records, scored, strict=True)
        if row.label == policy.requires_attention_label
    }
    false_clearance_trial_ids: set[str] = set()
    for record, row, family_index in zip(
        records,
        scored,
        predicted_families.tolist(),
        strict=True,
    ):
        family_threshold = family_thresholds[families[family_index]]
        if global_threshold is None or family_threshold is None:
            continue
        if row.score < max(global_threshold, family_threshold):
            continue
        if row.label == policy.requires_attention_label:
            false_clearances += 1
            false_clearance_trial_ids.add(record.scenario_id)
        else:
            true_clearances += 1
    cleared = false_clearances + true_clearances
    attention_trials = len(attention_trial_ids)
    false_clearance_trials = len(false_clearance_trial_ids)
    return {
        "threshold": global_threshold,
        "total": len(scored),
        "cleared": cleared,
        "true_clearances": true_clearances,
        "false_clearances": false_clearances,
        "attention_examples": attention_examples,
        "false_clearance_trials": false_clearance_trials,
        "attention_trials": attention_trials,
        "observed_false_clear_rate": (
            false_clearance_trials / attention_trials if attention_trials else 0.0
        ),
        "upper_false_clear_rate": exact_binomial_upper_bound(
            false_clearance_trials,
            attention_trials,
            policy.risk_control.confidence_level,
        ),
        "clearance_coverage": cleared / len(scored) if scored else 0.0,
    }


def _validate_serialized_boundary(
    model_dir: Path,
    records: list[DecisionRecord],
    scored: list[RiskExample],
    policy: PolicySpec,
    *,
    device: str,
    batch_size: int,
    limit: int = 128,
) -> None:
    """Reject an artifact if serving precision clears a high-scoring risky case."""
    from legal_decision_model.policy_inference import PolicyDecisionModel

    risky = sorted(
        (
            (row.score, record)
            for record, row in zip(records, scored, strict=True)
            if row.label == policy.requires_attention_label
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:limit]
    texts = [record.text for _, record in risky]
    model = PolicyDecisionModel(model_dir, device=device)
    individual = [model.predict(text) for text in texts]
    batched = model.predict_many(texts, batch_size=batch_size)
    failures = [
        record.record_id
        for (_, record), one, many in zip(risky, individual, batched, strict=True)
        if not one.requires_human_lawyer_attention or not many.requires_human_lawyer_attention
    ]
    if failures:
        preview = ", ".join(failures[:5])
        raise RuntimeError(
            f"serialized serving replay produced validation false clearances: {preview}"
        )
    print(
        f"Validated {len(risky):,} numerical-boundary attention records through "
        "serialized single and batched serving",
        flush=True,
    )


def train_policy_model(
    data_dir: Path = BENCHMARK_DIR,
    model_dir: Path = POLICY_MODEL_DIR,
    *,
    device: str = "mps",
    encoder_batch_size: int = 32,
    seed: int = POLICY_TRAINING_SEED,
    split_sizes: dict[str, int] | None = None,
    feature_cache_dir: Path = ARTIFACTS_DIR / "policy-feature-cache",
) -> PolicyTrainingSummary:
    """Train the Laya policy-head ensemble and risk controls."""
    print(f"Validating benchmark at {data_dir}", flush=True)
    validate_dataset(data_dir, split_sizes)
    policy = load_policy()
    families = tuple(spec.name for spec in FAMILIES)
    train_records = read_jsonl(data_dir / "train.jsonl")
    validation_records = read_jsonl(data_dir / "validation.jsonl")
    encoder = LayaFeatureEncoder(device=device, max_tokens=policy.model.max_tokens)
    train_cache = _cached_feature_cache(
        train_records,
        policy,
        encoder,
        families,
        data_dir / "train.jsonl",
        feature_cache_dir,
        encoder_batch_size,
    )
    feature_mean = train_cache.features.float().mean(dim=0)
    feature_scale = train_cache.features.float().std(dim=0).clamp(min=1e-4)
    train_cache = FeatureCache(
        _normalize(train_cache.features, feature_mean, feature_scale).to(torch.float16),
        train_cache.task_targets,
        train_cache.family_targets,
        train_cache.records,
    )
    config = PolicyHeadConfig(
        input_size=encoder.output_size,
        hidden_size=policy.model.hidden_size,
        task_names=task_names(policy),
        families=families,
        dropout=policy.model.dropout,
    )
    run_config = {
        "training_code_version": 8,
        "policy_name": policy.name,
        "policy_version": policy.version,
        "base_model_revision": BASE_MODEL_REVISION,
        "train_sha256": _sha256(data_dir / "train.jsonl"),
        "validation_sha256": _sha256(data_dir / "validation.jsonl"),
        "seed": seed,
        "model": asdict(policy.model),
        "risk_control": asdict(policy.risk_control),
        "head": {
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "task_names": list(config.task_names),
            "families": list(config.families),
            "dropout": config.dropout,
        },
    }
    work = model_dir.with_name(model_dir.name + ".work")
    if work.exists():
        config_path = work / "training-config.json"
        existing = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
        )
        if existing != run_config:
            shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "training-config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        heads = []
        for member in range(policy.model.ensemble_size):
            head_path = work / f"head-{member}.safetensors"
            if head_path.is_file():
                print(
                    f"Loading completed ensemble member {member + 1} from {head_path}",
                    flush=True,
                )
                head = load_head(head_path, config, encoder.device)
            else:
                head = _train_member(
                    train_cache,
                    config,
                    policy,
                    member_index=member,
                    seed=seed + member,
                    device=encoder.device,
                )
                temporary = work / f"head-{member}.safetensors.tmp"
                save_head(temporary, head)
                temporary.replace(head_path)
            heads.append(head)
        train_centroids = []
        family_centroids: dict[str, tuple[float, ...]] = {}
        train_in_scope = torch.tensor(
            [record.in_scope for record in train_records],
            dtype=torch.bool,
        )
        for family_index, family in enumerate(families):
            mask = (train_cache.family_targets == family_index) & train_in_scope
            centroid = train_cache.features[mask].float().mean(dim=0)
            train_centroids.append(centroid)
            family_centroids[family] = tuple(float(value) for value in centroid)
        centroids = torch.stack(train_centroids)
        validation_documents = _cached_document_features(
            validation_records,
            policy,
            encoder,
            data_dir / "validation.jsonl",
            feature_cache_dir,
            encoder_batch_size,
        )
        validation_features = _normalize(
            validation_documents.features,
            feature_mean,
            feature_scale,
        ).to(torch.float16)
        (
            task_probabilities,
            family_logits,
            task_disagreement,
            validation_distance,
        ) = _document_outputs(
            heads,
            validation_features,
            validation_documents.window_counts,
            centroids,
            policy,
            config.task_names,
            encoder.device,
        )
        in_scope_mask = torch.tensor(
            [record.in_scope for record in validation_records],
            dtype=torch.bool,
        )
        ood_threshold = quantile(validation_distance[in_scope_mask], 0.995)
        disagreement_threshold = quantile(
            task_disagreement[in_scope_mask].max(dim=1).values,
            0.995,
        )
        scored = _safe_scores(
            task_probabilities,
            task_disagreement,
            tuple(validation_records),
            policy,
            config.task_names,
            validation_distance,
            ood_threshold,
            disagreement_threshold,
        )
        global_selection = select_risk_controlled_threshold(
            scored,
            maximum_false_clear_rate=policy.risk_control.maximum_false_clear_rate,
            confidence_level=policy.risk_control.confidence_level,
            maximum_false_clearances=(policy.risk_control.maximum_observed_false_clearances),
        )
        predicted_families = family_logits.argmax(dim=1)
        validation_family_targets = torch.tensor(
            [families.index(record.family) for record in validation_records],
            dtype=torch.long,
        )
        family_thresholds: dict[str, float | None] = {}
        family_reports: dict[str, object] = {}
        for family_index, family in enumerate(families):
            indices = [
                index
                for index, predicted in enumerate(predicted_families.tolist())
                if predicted == family_index
            ]
            group_rows = [scored[index] for index in indices]
            attention_count = len(
                {row.trial_id for row in group_rows if row.label == policy.requires_attention_label}
            )
            if (
                not group_rows
                or attention_count < policy.risk_control.minimum_group_attention_examples
            ):
                family_thresholds[family] = None
                family_reports[family] = {
                    "status": "insufficient_attention_examples",
                    "attention_examples": attention_count,
                }
                continue
            selection = select_risk_controlled_threshold(
                group_rows,
                maximum_false_clear_rate=(policy.risk_control.maximum_group_false_clear_rate),
                confidence_level=policy.risk_control.confidence_level,
                maximum_false_clearances=(policy.risk_control.maximum_observed_false_clearances),
            )
            family_thresholds[family] = selection.threshold
            family_reports[family] = _selection_dict(selection)
        validation_summary = {
            **_serving_validation_summary(
                scored,
                validation_records,
                predicted_families,
                families,
                global_selection.threshold,
                family_thresholds,
                policy,
            ),
            "global_calibration_coverage": global_selection.point.clearance_coverage,
            "ood_distance_threshold": ood_threshold,
            "ensemble_disagreement_threshold": disagreement_threshold,
            "family_accuracy": float(
                (predicted_families == validation_family_targets).float().mean()
            ),
        }
        metadata = PolicyModelMetadata(
            format_version=3,
            policy_name=policy.name,
            policy_version=policy.version,
            policy_fingerprint=policy_fingerprint(policy),
            base_model=BASE_MODEL,
            base_model_revision=BASE_MODEL_REVISION,
            task_names=config.task_names,
            families=families,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            ensemble_size=len(heads),
            global_threshold=global_selection.threshold,
            family_thresholds=family_thresholds,
            confidence_level=policy.risk_control.confidence_level,
            maximum_false_clear_rate=policy.risk_control.maximum_false_clear_rate,
            maximum_observed_false_clearances=(
                policy.risk_control.maximum_observed_false_clearances
            ),
            maximum_group_false_clear_rate=(policy.risk_control.maximum_group_false_clear_rate),
            minimum_group_attention_examples=(policy.risk_control.minimum_group_attention_examples),
            ood_distance_threshold=ood_threshold,
            ensemble_disagreement_threshold=disagreement_threshold,
            feature_mean=tuple(float(value) for value in feature_mean),
            feature_scale=tuple(float(value) for value in feature_scale),
            family_centroids=family_centroids,
            max_tokens=policy.model.max_tokens,
            window_overlap=policy.model.window_overlap,
            maximum_windows=policy.model.maximum_windows,
            training_rows=len(train_records),
            validation_rows=len(validation_records),
            trained_at=datetime.now(UTC).isoformat(),
            validation_summary=validation_summary,
        )
        save_policy_metadata(work / METADATA_FILE, metadata)
        (work / RISK_REPORT_FILE).write_text(
            json.dumps(
                {
                    "global": _selection_dict(global_selection),
                    "families": family_reports,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        del heads
        del encoder
        if device == "mps":
            torch.mps.empty_cache()
        _validate_serialized_boundary(
            work,
            validation_records,
            scored,
            policy,
            device=device,
            batch_size=encoder_batch_size,
        )
        if model_dir.exists():
            shutil.rmtree(model_dir)
        work.replace(model_dir)
        print(f"Saved policy ensemble to {model_dir}", flush=True)
    except BaseException:
        print(f"Retained resumable training state at {work}", flush=True)
        raise
    return PolicyTrainingSummary(model_dir, metadata)
