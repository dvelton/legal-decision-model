"""Load and run the trained decision head."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from laya.common import build_sequence
from stuntd.serve.decider import Decider
from stuntd.train.artifacts import SiteModel
from stuntd.train.trainer import question_for

from legal_decision_model.calibration import decide
from legal_decision_model.constants import MODEL_DIR, NO_ATTENTION, REQUIRES_ATTENTION
from legal_decision_model.model_artifacts import HEAD_FILE, ModelMetadata, load_metadata
from legal_decision_model.revisions import pinned_laya_revision

MAX_INPUT_CHARACTERS = 8_000


@dataclass(frozen=True)
class Decision:
    """Public inference result."""

    decision: str
    requires_human_lawyer_attention: bool
    probability_no_human_lawyer_attention: float
    probability_requires_human_lawyer_attention: float
    clearance_threshold: float | None
    routed_conservatively: bool
    routing_reason: str
    latency_ms: int
    model: str
    policy_version: str
    disclaimer: str = (
        "Demonstration output from a model trained only on synthetic data. "
        "It is not legal advice and is not validated for real legal triage."
    )

    def to_dict(self) -> dict[str, str | bool | int | float | None]:
        """Return a JSON-serializable result."""
        return asdict(self)


class LegalDecisionModel:
    """One loaded base encoder and its task-specific legal-attention head."""

    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "auto") -> None:
        self.model_dir = model_dir
        self.metadata = load_metadata(model_dir)
        self.head_path = model_dir / HEAD_FILE
        if not self.head_path.is_file():
            raise FileNotFoundError(f"trained head not found at {self.head_path}")
        with pinned_laya_revision(self.metadata.base_model_revision):
            self._decider = Decider(self.metadata.base_model, device=device, lazy=False)
        self._tokenizer: Any = self._decider._agent.tok
        self._site_model = _site_model(self.metadata)
        self._question = question_for(
            self.metadata.field,
            self.metadata.labels,
            self.metadata.spaced_labels,
        )

    def predict(self, text: str) -> Decision:
        """Return the conservative binary routing decision."""
        normalized = text.strip()
        if not normalized:
            raise ValueError("request text must not be empty")
        if len(normalized) > MAX_INPUT_CHARACTERS:
            raise ValueError(
                f"request text exceeds the {MAX_INPUT_CHARACTERS}-character demonstration limit"
            )
        if self._would_truncate(normalized):
            return Decision(
                decision=REQUIRES_ATTENTION,
                requires_human_lawyer_attention=True,
                probability_no_human_lawyer_attention=0.0,
                probability_requires_human_lawyer_attention=1.0,
                clearance_threshold=self.metadata.clearance_threshold,
                routed_conservatively=True,
                routing_reason="INPUT_EXCEEDS_MODEL_TOKEN_BUDGET",
                latency_ms=0,
                model=self.metadata.base_model,
                policy_version=self.metadata.policy_version,
            )
        verdict = self._decider.decide(self._site_model, self.head_path, normalized)
        probability_safe = verdict.probabilities[0]
        label = decide(probability_safe, self.metadata.clearance_threshold)
        return Decision(
            decision=label,
            requires_human_lawyer_attention=label == REQUIRES_ATTENTION,
            probability_no_human_lawyer_attention=probability_safe,
            probability_requires_human_lawyer_attention=1.0 - probability_safe,
            clearance_threshold=self.metadata.clearance_threshold,
            routed_conservatively=(
                label == REQUIRES_ATTENTION
                and verdict.label == self.metadata.labels.index(NO_ATTENTION)
            ),
            routing_reason="MODEL_DECISION",
            latency_ms=verdict.latency_ms,
            model=self.metadata.base_model,
            policy_version=self.metadata.policy_version,
        )

    def _would_truncate(self, text: str) -> bool:
        _, _, stats = build_sequence(
            self._tokenizer,
            text,
            self._question,
            self.metadata.max_len,
            self.metadata.head_max_len,
            return_truncation_stats=True,
        )
        return bool(stats["truncated"])


def _site_model(metadata: ModelMetadata) -> SiteModel:
    return SiteModel(
        site=metadata.site,
        kind="choice",
        field=metadata.field,
        labels=list(metadata.labels),
        base_model=metadata.base_model,
        temperature=metadata.temperature,
        threshold=metadata.clearance_threshold,
        target_agreement=1.0 - metadata.max_false_clear_rate,
        trained_at=0.0,
        n_train=metadata.training_rows,
        n_holdout=metadata.validation_rows,
        agreement=float(metadata.validation_metrics["accuracy"]),
        coverage=float(metadata.validation_metrics["clearance_coverage"]),
        covered_agreement=1.0 - float(metadata.validation_metrics["clearance_error_rate"]),
        ece=0.0,
        per_class={},
        confident_errors=[],
        curve=[],
        max_len=metadata.max_len,
        head_max_len=metadata.head_max_len,
        spaced_labels=metadata.spaced_labels,
    )
