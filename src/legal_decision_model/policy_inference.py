"""Conservative inference for the multi-task Laya policy ensemble."""

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from legal_decision_model.constants import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    POLICY_MODEL_DIR,
)
from legal_decision_model.policy_model import (
    FINAL_TASK,
    IN_SCOPE_TASK,
    LayaFeatureEncoder,
    PolicyHeadConfig,
    PolicyModelMetadata,
    cosine_distance,
    load_head,
    load_policy_metadata,
    policy_model_fingerprint,
    safe_score,
    split_text_windows,
)
from legal_decision_model.policy_schema import PolicySpec, load_policy, policy_fingerprint
from legal_decision_model.policy_training import METADATA_FILE


@dataclass(frozen=True)
class PolicyDecision:
    """Public decision with auditable internal policy signals."""

    decision: str
    requires_human_lawyer_attention: bool
    clearance_score: float
    clearance_threshold: float | None
    workflow_family: str
    routing_reason: str
    reason_codes: tuple[str, ...]
    task_probabilities: dict[str, float]
    ensemble_disagreement: float
    ood_distance: float
    routed_conservatively: bool
    token_count: int
    window_count: int
    latency_ms: int
    model: str
    policy_version: str
    disclaimer: str = (
        "Demonstration output from a model trained only on synthetic data. "
        "It is not legal advice and is not validated for real legal triage."
    )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result."""
        return asdict(self)


class PolicyDecisionModel:
    """One Laya encoder shared by a multi-task policy-head ensemble."""

    def __init__(
        self,
        model_dir: Path = POLICY_MODEL_DIR,
        *,
        device: str = "auto",
    ) -> None:
        self.model_dir = model_dir
        self.metadata = load_policy_metadata(model_dir / METADATA_FILE)
        self.policy = load_policy()
        self._validate_artifact()
        self.encoder = LayaFeatureEncoder(
            device=device,
            max_tokens=self.metadata.max_tokens,
        )
        if self.encoder.output_size != self.metadata.input_size:
            raise ValueError("policy-model input size does not match the Laya encoder")
        config = PolicyHeadConfig(
            input_size=self.metadata.input_size,
            hidden_size=self.metadata.hidden_size,
            task_names=self.metadata.task_names,
            families=self.metadata.families,
            dropout=self.metadata.dropout,
        )
        self.heads = [
            load_head(
                model_dir / f"head-{member}.safetensors",
                config,
                self.encoder.device,
            )
            for member in range(self.metadata.ensemble_size)
        ]
        self.artifact_fingerprint = policy_model_fingerprint(self.metadata, self.heads)
        self.feature_mean = torch.tensor(self.metadata.feature_mean)
        self.feature_scale = torch.tensor(self.metadata.feature_scale)
        self.centroids = torch.tensor(
            [self.metadata.family_centroids[family] for family in self.metadata.families]
        )

    def predict(self, text: str) -> PolicyDecision:
        """Return the risk-controlled binary routing decision."""
        started = time.perf_counter()
        normalized = text.strip()
        if not normalized:
            raise ValueError("request text must not be empty")
        token_ids = self.encoder.token_ids(normalized)
        windows = self._windows(normalized, token_ids)
        if windows is None:
            return self._conservative_result(
                "INPUT_EXCEEDS_MAXIMUM_WINDOWS",
                len(token_ids),
                self.metadata.maximum_windows + 1,
                started,
            )
        raw_features = self.encoder.encode(windows)
        return self._decision_from_features(
            raw_features,
            len(token_ids),
            len(windows),
            started,
        )

    def predict_many(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
    ) -> tuple[PolicyDecision, ...]:
        """Run many requests through one batched encoder pass."""
        started = time.perf_counter()
        plans: list[tuple[int, int, int] | None] = []
        flat_windows: list[str] = []
        over_budget_tokens: list[int] = []
        for text in texts:
            normalized = text.strip()
            if not normalized:
                raise ValueError("request text must not be empty")
            token_ids = self.encoder.token_ids(normalized)
            windows = self._windows(normalized, token_ids)
            if windows is None:
                plans.append(None)
                over_budget_tokens.append(len(token_ids))
                continue
            start = len(flat_windows)
            flat_windows.extend(windows)
            plans.append((start, len(windows), len(token_ids)))
            over_budget_tokens.append(0)
        raw_features = (
            self.encoder.encode(flat_windows, batch_size=batch_size)
            if flat_windows
            else torch.empty((0, self.metadata.input_size), dtype=torch.float16)
        )
        results = []
        for index, plan in enumerate(plans):
            if plan is None:
                results.append(
                    self._conservative_result(
                        "INPUT_EXCEEDS_MAXIMUM_WINDOWS",
                        over_budget_tokens[index],
                        self.metadata.maximum_windows + 1,
                        started,
                    )
                )
                continue
            start, window_count, token_count = plan
            results.append(
                self._decision_from_features(
                    raw_features[start : start + window_count],
                    token_count,
                    window_count,
                    started,
                )
            )
        return tuple(results)

    def _decision_from_features(
        self,
        raw_features: torch.Tensor,
        token_count: int,
        window_count: int,
        started: float,
    ) -> PolicyDecision:
        if not torch.isfinite(raw_features).all():
            return self._conservative_result(
                "NON_FINITE_MODEL_OUTPUT",
                token_count,
                window_count,
                started,
            )
        features = ((raw_features.float() - self.feature_mean) / self.feature_scale).to(
            torch.float16
        )
        if not torch.isfinite(features).all():
            return self._conservative_result(
                "NON_FINITE_MODEL_OUTPUT",
                token_count,
                window_count,
                started,
            )
        member_tasks, member_families = self._member_outputs(features)
        if not torch.isfinite(member_tasks).all() or not torch.isfinite(member_families).all():
            return self._conservative_result(
                "NON_FINITE_MODEL_OUTPUT",
                token_count,
                window_count,
                started,
            )
        task_probabilities = member_tasks.mean(dim=0)
        family_logits = member_families.mean(dim=(0, 1))
        family_index = int(family_logits.argmax())
        family = self.metadata.families[family_index]
        aggregated = self._aggregate_tasks(task_probabilities)
        score = safe_score(aggregated, self.policy)
        disagreement = float(member_tasks.std(dim=0, correction=0).max())
        distances = cosine_distance(features, self.centroids)
        ood_distance = float(distances.max())
        if not all(
            math.isfinite(value)
            for value in (
                score,
                disagreement,
                ood_distance,
                *aggregated.values(),
            )
        ):
            return self._conservative_result(
                "NON_FINITE_MODEL_OUTPUT",
                token_count,
                window_count,
                started,
            )
        threshold = self._threshold(family)
        reason_codes = self._reason_codes(aggregated)
        routed_conservatively = False
        if ood_distance > self.metadata.ood_distance_threshold:
            requires_attention = True
            routing_reason = "OUT_OF_DISTRIBUTION"
            routed_conservatively = True
        elif disagreement > self.metadata.ensemble_disagreement_threshold:
            requires_attention = True
            routing_reason = "ENSEMBLE_DISAGREEMENT"
            routed_conservatively = True
        elif threshold is None:
            requires_attention = True
            routing_reason = "CLEARANCE_DISABLED"
            routed_conservatively = True
        elif score < threshold:
            requires_attention = True
            routing_reason = "RISK_CONTROL_THRESHOLD"
            routed_conservatively = aggregated[FINAL_TASK] < 0.5
        else:
            requires_attention = False
            routing_reason = "MODEL_CLEARANCE"
            reason_codes = (self.policy.success_reason,)
        decision = (
            self.policy.requires_attention_label
            if requires_attention
            else self.policy.no_attention_label
        )
        return PolicyDecision(
            decision=decision,
            requires_human_lawyer_attention=requires_attention,
            clearance_score=score,
            clearance_threshold=threshold,
            workflow_family=family,
            routing_reason=routing_reason,
            reason_codes=reason_codes,
            task_probabilities=aggregated,
            ensemble_disagreement=disagreement,
            ood_distance=ood_distance,
            routed_conservatively=routed_conservatively,
            token_count=token_count,
            window_count=window_count,
            latency_ms=round((time.perf_counter() - started) * 1000),
            model=self.metadata.base_model,
            policy_version=self.metadata.policy_version,
        )

    def _member_outputs(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        task_outputs = []
        family_outputs = []
        with torch.no_grad():
            device_features = features.to(self.encoder.device, torch.float32)
            for head in self.heads:
                task_logits, family_logits = head(device_features)
                task_outputs.append(torch.sigmoid(task_logits).cpu())
                family_outputs.append(family_logits.cpu())
        return torch.stack(task_outputs), torch.stack(family_outputs)

    def _aggregate_tasks(self, probabilities: torch.Tensor) -> dict[str, float]:
        output = {}
        prerequisite_set = set(self.policy.prerequisites)
        for position, name in enumerate(self.metadata.task_names):
            values = probabilities[:, position]
            if name in prerequisite_set or name == IN_SCOPE_TASK:
                output[name] = float(values.min())
            else:
                output[name] = float(values.max())
        return output

    def _windows(self, text: str, token_ids: list[int]) -> list[str] | None:
        return split_text_windows(
            self.encoder,
            text,
            max_tokens=self.metadata.max_tokens,
            window_overlap=self.metadata.window_overlap,
            maximum_windows=self.metadata.maximum_windows,
            token_ids=token_ids,
        )

    def _threshold(self, family: str) -> float | None:
        global_threshold = self.metadata.global_threshold
        family_threshold = self.metadata.family_thresholds[family]
        if global_threshold is None or family_threshold is None:
            return None
        return max(global_threshold, family_threshold)

    def _reason_codes(
        self,
        probabilities: dict[str, float],
    ) -> tuple[str, ...]:
        reasons = [
            self.policy.prerequisite_reasons[name]
            for name in self.policy.prerequisites
            if probabilities[name] < 0.5
        ]
        reasons.extend(
            self.policy.trigger_reasons[name]
            for name in self.policy.triggers
            if probabilities[name] >= 0.5
        )
        if probabilities[IN_SCOPE_TASK] < 0.5:
            reasons.append("REQUEST_OUTSIDE_POLICY_SCOPE")
        if not reasons:
            reasons.append("MODEL_REQUIRES_HUMAN_ATTENTION")
        return tuple(reasons)

    def _conservative_result(
        self,
        routing_reason: str,
        token_count: int,
        window_count: int,
        started: float,
    ) -> PolicyDecision:
        return PolicyDecision(
            decision=self.policy.requires_attention_label,
            requires_human_lawyer_attention=True,
            clearance_score=0.0,
            clearance_threshold=self.metadata.global_threshold,
            workflow_family="unknown",
            routing_reason=routing_reason,
            reason_codes=(routing_reason,),
            task_probabilities={},
            ensemble_disagreement=0.0,
            ood_distance=0.0,
            routed_conservatively=True,
            token_count=token_count,
            window_count=window_count,
            latency_ms=round((time.perf_counter() - started) * 1000),
            model=self.metadata.base_model,
            policy_version=self.metadata.policy_version,
        )

    def _validate_artifact(self) -> None:
        metadata: PolicyModelMetadata = self.metadata
        policy: PolicySpec = self.policy
        if metadata.format_version != 3:
            raise ValueError(f"unsupported policy-model format {metadata.format_version}")
        if metadata.policy_name != policy.name or metadata.policy_version != policy.version:
            raise ValueError("policy-model artifact does not match the configured policy")
        if metadata.policy_fingerprint != policy_fingerprint(policy):
            raise ValueError("policy-model artifact does not match the configured policy settings")
        if metadata.base_model != BASE_MODEL or metadata.base_model_revision != BASE_MODEL_REVISION:
            raise ValueError("policy-model base model does not match the pinned model")
        if metadata.task_names != (
            *policy.prerequisites,
            *policy.triggers,
            IN_SCOPE_TASK,
            FINAL_TASK,
        ):
            raise ValueError("policy-model task order does not match the policy")
        if (
            metadata.ensemble_size != policy.model.ensemble_size
            or metadata.hidden_size != policy.model.hidden_size
            or metadata.dropout != policy.model.dropout
            or metadata.max_tokens != policy.model.max_tokens
            or metadata.window_overlap != policy.model.window_overlap
            or metadata.maximum_windows != policy.model.maximum_windows
            or metadata.maximum_observed_false_clearances
            != policy.risk_control.maximum_observed_false_clearances
            or metadata.confidence_level != policy.risk_control.confidence_level
            or metadata.maximum_false_clear_rate != policy.risk_control.maximum_false_clear_rate
            or metadata.maximum_group_false_clear_rate
            != policy.risk_control.maximum_group_false_clear_rate
            or metadata.minimum_group_attention_examples
            != policy.risk_control.minimum_group_attention_examples
        ):
            raise ValueError("policy-model architecture does not match the policy")
        for member in range(metadata.ensemble_size):
            path = self.model_dir / f"head-{member}.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"policy-model head not found at {path}")
