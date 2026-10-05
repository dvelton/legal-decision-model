"""Laya-encoder multi-task policy model."""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import laya
import torch
from laya.common import encode_text
from safetensors.torch import load_file, save_file
from torch import nn

from legal_decision_model.constants import BASE_MODEL, BASE_MODEL_REVISION
from legal_decision_model.policy_schema import PolicySpec
from legal_decision_model.records import DecisionRecord
from legal_decision_model.revisions import pinned_laya_revision

FINAL_TASK = "requires_human_lawyer_attention"
IN_SCOPE_TASK = "in_scope"


def task_names(policy: PolicySpec) -> tuple[str, ...]:
    """Stable multi-task output order."""
    return (*policy.prerequisites, *policy.triggers, IN_SCOPE_TASK, FINAL_TASK)


def task_targets(record: DecisionRecord, policy: PolicySpec) -> list[float]:
    """Targets aligned to task_names."""
    facts = policy.validate_facts(record.facts)
    return [
        *[float(facts[name]) for name in policy.prerequisites],
        *[float(facts[name]) for name in policy.triggers],
        float(record.in_scope),
        float(record.label == policy.requires_attention_label),
    ]


def safe_score(probabilities: dict[str, float], policy: PolicySpec) -> float:
    """Conservative probability that every clearance condition holds."""
    components = [
        *[probabilities[name] for name in policy.prerequisites],
        *[1.0 - probabilities[name] for name in policy.triggers],
        probabilities[IN_SCOPE_TASK],
        1.0 - probabilities[FINAL_TASK],
    ]
    if any(not math.isfinite(component) for component in components):
        raise ValueError("policy probabilities must be finite")
    return min(components)


@dataclass(frozen=True)
class PolicyHeadConfig:
    """Serialized architecture description."""

    input_size: int
    hidden_size: int
    task_names: tuple[str, ...]
    families: tuple[str, ...]
    dropout: float


class PolicyHead(nn.Module):
    """Small shared trunk with policy-task and family outputs."""

    def __init__(self, config: PolicyHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = nn.Sequential(
            nn.LayerNorm(config.input_size),
            nn.Linear(config.input_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.tasks = nn.Linear(config.hidden_size, len(config.task_names))
        self.family = nn.Linear(config.hidden_size, len(config.families))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.tasks(hidden), self.family(hidden)


@dataclass(frozen=True)
class FeatureCache:
    """Encoded records and supervised targets."""

    features: torch.Tensor
    task_targets: torch.Tensor
    family_targets: torch.Tensor
    records: tuple[DecisionRecord, ...]


class LayaFeatureEncoder:
    """Frozen Laya encoder used by every policy head."""

    def __init__(
        self,
        device: str = "auto",
        max_tokens: int = 512,
    ) -> None:
        with pinned_laya_revision(BASE_MODEL_REVISION):
            self.agent: Any = laya.Agent(
                BASE_MODEL,
                device=None if device == "auto" else device,
            )
        self.agent.model.eval()
        self.tokenizer: Any = self.agent.tok
        self.encoder: Any = self.agent.model.encoder
        self.device: torch.device = self.agent.device
        self.max_tokens = max_tokens
        self.output_size = int(self.encoder.config.hidden_size)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Mean-pool non-padding token representations."""
        encoded: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = list(texts[start : start + batch_size])
                batch = encode_text(
                    self.tokenizer,
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                )
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                hidden = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                weights = attention_mask[:, :, None].to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
                encoded.append(pooled.to("cpu", torch.float16))
        return torch.cat(encoded)

    def token_ids(self, text: str) -> list[int]:
        """Tokenize without truncation for window construction."""
        values = encode_text(
            self.tokenizer,
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        return list(values)

    def decode_tokens(self, ids: Sequence[int]) -> str:
        """Decode one token window back to normalized text."""
        return str(self.tokenizer.decode(list(ids), skip_special_tokens=True))


def split_text_windows(
    encoder: Any,
    text: str,
    *,
    max_tokens: int,
    window_overlap: int,
    maximum_windows: int,
    token_ids: list[int] | None = None,
) -> list[str] | None:
    """Split text into the same overlapping token windows used in serving."""
    if token_ids is None:
        token_ids = encoder.token_ids(text)
    special_tokens = int(encoder.tokenizer.num_special_tokens_to_add(pair=False))
    window_size = max_tokens - special_tokens
    if window_size < 1:
        raise ValueError("model token budget cannot contain input text")
    if len(token_ids) <= window_size:
        return [text]
    step = window_size - window_overlap
    if step < 1:
        raise ValueError("window overlap must be below usable token capacity")
    starts = list(range(0, len(token_ids) - window_size + 1, step))
    final_start = len(token_ids) - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    if len(starts) > maximum_windows:
        return None
    return [encoder.decode_tokens(token_ids[start : start + window_size]) for start in starts]


def build_feature_cache(
    records: Sequence[DecisionRecord],
    policy: PolicySpec,
    encoder: LayaFeatureEncoder,
    families: Sequence[str],
    batch_size: int = 32,
) -> FeatureCache:
    """Encode records and align task and family targets."""
    family_index = {name: index for index, name in enumerate(families)}
    return FeatureCache(
        features=encoder.encode([record.text for record in records], batch_size=batch_size),
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


def save_head(path: Path, head: PolicyHead) -> None:
    """Save one ensemble member in portable float32 safetensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in head.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("policy-model head contains non-finite values")
    save_file(state, str(path))


def load_head(path: Path, config: PolicyHeadConfig, device: torch.device) -> PolicyHead:
    """Load one ensemble member."""
    head = PolicyHead(config)
    state = load_file(str(path))
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError(f"policy-model head contains non-finite values: {path}")
    head.load_state_dict(state)
    head.to(device)
    head.eval()
    return head


@dataclass(frozen=True)
class PolicyModelMetadata:
    """Persisted multi-task ensemble and risk-control metadata."""

    format_version: int
    policy_name: str
    policy_version: str
    policy_fingerprint: str
    base_model: str
    base_model_revision: str
    task_names: tuple[str, ...]
    families: tuple[str, ...]
    input_size: int
    hidden_size: int
    dropout: float
    ensemble_size: int
    global_threshold: float | None
    family_thresholds: dict[str, float | None]
    confidence_level: float
    maximum_false_clear_rate: float
    maximum_observed_false_clearances: int
    maximum_group_false_clear_rate: float
    minimum_group_attention_examples: int
    ood_distance_threshold: float
    ensemble_disagreement_threshold: float
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    family_centroids: dict[str, tuple[float, ...]]
    max_tokens: int
    window_overlap: int
    maximum_windows: int
    training_rows: int
    validation_rows: int
    trained_at: str
    validation_summary: dict[str, float | int | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PolicyModelMetadata":
        values = dict(raw)
        for name in ("task_names", "families", "feature_mean", "feature_scale"):
            values[name] = tuple(values[name])
        values["family_centroids"] = {
            name: tuple(centroid) for name, centroid in values["family_centroids"].items()
        }
        metadata = cls(**values)
        if metadata.format_version != 3:
            raise ValueError(f"unsupported policy-model format {metadata.format_version}")
        if metadata.ensemble_size < 1:
            raise ValueError("policy-model ensemble must contain at least one head")
        if metadata.input_size < 1 or metadata.hidden_size < 1:
            raise ValueError("policy-model dimensions must be positive")
        if not 0 <= metadata.dropout < 1:
            raise ValueError("policy-model dropout must be between zero and one")
        bounded_rates = (
            metadata.confidence_level,
            metadata.maximum_false_clear_rate,
            metadata.maximum_group_false_clear_rate,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in bounded_rates):
            raise ValueError("policy-model confidence and risk rates must be within [0, 1]")
        if metadata.maximum_observed_false_clearances < 0:
            raise ValueError("policy-model observed false-clear limit must be non-negative")
        if metadata.minimum_group_attention_examples < 1:
            raise ValueError("policy-model minimum group evidence must be positive")
        if len(metadata.policy_fingerprint) != 64:
            raise ValueError("policy-model policy fingerprint is invalid")
        if (
            len(metadata.feature_mean) != metadata.input_size
            or len(metadata.feature_scale) != metadata.input_size
        ):
            raise ValueError("policy-model feature statistics have the wrong size")
        finite_vectors = [
            *metadata.feature_mean,
            *metadata.feature_scale,
            *(value for centroid in metadata.family_centroids.values() for value in centroid),
        ]
        if any(not math.isfinite(value) for value in finite_vectors):
            raise ValueError("policy-model feature statistics must be finite")
        if any(value <= 0 for value in metadata.feature_scale):
            raise ValueError("policy-model feature scales must be positive")
        if set(metadata.family_centroids) != set(metadata.families):
            raise ValueError("policy-model family centroids do not match its families")
        if set(metadata.family_thresholds) != set(metadata.families):
            raise ValueError("policy-model family thresholds do not match its families")
        if any(
            len(centroid) != metadata.input_size for centroid in metadata.family_centroids.values()
        ):
            raise ValueError("policy-model family centroid has the wrong size")
        thresholds = [
            metadata.global_threshold,
            *metadata.family_thresholds.values(),
        ]
        if any(
            threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1)
            for threshold in thresholds
        ):
            raise ValueError("policy-model clearance threshold is outside [0, 1]")
        if (
            not math.isfinite(metadata.ood_distance_threshold)
            or metadata.ood_distance_threshold < 0
        ):
            raise ValueError("policy-model OOD threshold must be non-negative")
        if (
            not math.isfinite(metadata.ensemble_disagreement_threshold)
            or not 0 <= metadata.ensemble_disagreement_threshold <= 1
        ):
            raise ValueError("policy-model disagreement threshold is outside [0, 1]")
        if metadata.max_tokens < 1 or metadata.maximum_windows < 1:
            raise ValueError("policy-model token and window limits must be positive")
        if not 0 <= metadata.window_overlap < metadata.max_tokens:
            raise ValueError("policy-model window overlap is invalid")
        if any(
            isinstance(value, (int, float)) and not math.isfinite(float(value))
            for value in metadata.validation_summary.values()
        ):
            raise ValueError("policy-model validation summary must be finite")
        return metadata


def save_policy_metadata(path: Path, metadata: PolicyModelMetadata) -> None:
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_policy_metadata(path: Path) -> PolicyModelMetadata:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object")
    return PolicyModelMetadata.from_dict(raw)


def policy_model_fingerprint(
    metadata: PolicyModelMetadata,
    heads: Sequence[PolicyHead],
) -> str:
    """Hash the exact in-memory policy model used for decisions."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    for member, head in enumerate(heads):
        digest.update(str(member).encode("ascii"))
        for name, value in sorted(head.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def standardized_distance(
    features: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Root-mean-square standardized embedding distance."""
    return torch.sqrt(torch.mean(((features.float() - mean) / scale) ** 2, dim=1))


def cosine_distance(
    features: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Distance to the nearest family centroid."""
    normalized_features = nn.functional.normalize(features.float(), dim=1)
    normalized_centroids = nn.functional.normalize(centroids.float(), dim=1)
    similarities = normalized_features @ normalized_centroids.T
    return 1.0 - similarities.max(dim=1).values


def quantile(values: torch.Tensor, probability: float) -> float:
    """Stable scalar quantile."""
    if values.numel() == 0:
        raise ValueError("cannot take a quantile of no values")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if not torch.isfinite(values).all():
        raise ValueError("cannot take a quantile of non-finite values")
    sorted_values = torch.sort(values.float()).values
    index = min(
        len(sorted_values) - 1,
        max(0, math.ceil(probability * len(sorted_values)) - 1),
    )
    return float(sorted_values[index])
