"""Versioned policy schema and compiler."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from legal_decision_model.constants import DEFAULT_POLICY_PATH


@dataclass(frozen=True)
class RiskControlSpec:
    """Statistical operating constraints."""

    confidence_level: float
    maximum_false_clear_rate: float
    maximum_observed_false_clearances: int
    maximum_group_false_clear_rate: float
    minimum_group_attention_examples: int


@dataclass(frozen=True)
class ModelSpec:
    """Policy-model training and inference configuration."""

    ensemble_size: int
    hidden_size: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    ranking_margin: float
    ranking_weight: float
    consistency_weight: float
    max_tokens: int
    window_overlap: int
    maximum_windows: int


@dataclass(frozen=True)
class PolicySpec:
    """Compiled executable policy."""

    name: str
    version: str
    description: str
    requires_attention_label: str
    no_attention_label: str
    prerequisites: tuple[str, ...]
    prerequisite_defaults: dict[str, bool]
    prerequisite_reasons: dict[str, str]
    triggers: tuple[str, ...]
    trigger_reasons: dict[str, str]
    success_reason: str
    risk_control: RiskControlSpec
    model: ModelSpec

    @property
    def fact_names(self) -> tuple[str, ...]:
        """Every structured fact in stable model-output order."""
        return (*self.prerequisites, *self.triggers)

    def default_facts(self) -> dict[str, bool]:
        """Return a safe default fact set."""
        return {
            **self.prerequisite_defaults,
            **dict.fromkeys(self.triggers, False),
        }

    def validate_facts(self, facts: dict[str, Any]) -> dict[str, bool]:
        """Refuse missing, unknown, or non-boolean policy facts."""
        expected = set(self.fact_names)
        unknown = set(facts) - expected
        missing = expected - set(facts)
        if unknown:
            raise ValueError(f"unknown policy facts: {', '.join(sorted(unknown))}")
        if missing:
            raise ValueError(f"missing policy facts: {', '.join(sorted(missing))}")
        if any(not isinstance(value, bool) for value in facts.values()):
            raise ValueError("every policy fact must be a boolean")
        return {name: facts[name] for name in self.fact_names}


def policy_fingerprint(policy: PolicySpec) -> str:
    """Hash every compiled setting that affects generation, training, or serving."""
    payload = json.dumps(
        asdict(policy),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(raw: object, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{where} keys must be strings")
    return dict(raw)


def _string(raw: object, where: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return raw.strip()


def _number(raw: object, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"{where} must be a number")
    return float(raw)


def _integer(raw: object, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{where} must be an integer")
    return raw


def _bounded(value: float, where: str, low: float, high: float) -> float:
    if not low <= value <= high:
        raise ValueError(f"{where} must be between {low} and {high}")
    return value


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> PolicySpec:
    """Load and validate a policy YAML file."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"policy not found at {path}") from exc
    raw = _mapping(loaded, "policy")
    labels = _mapping(raw.get("labels"), "labels")
    prerequisites = _mapping(raw.get("prerequisites"), "prerequisites")
    triggers = _mapping(raw.get("triggers"), "triggers")
    clearance = _mapping(raw.get("clearance"), "clearance")
    risk = _mapping(raw.get("risk_control"), "risk_control")
    model = _mapping(raw.get("model"), "model")
    prerequisite_defaults: dict[str, bool] = {}
    prerequisite_reasons: dict[str, str] = {}
    for name, definition in prerequisites.items():
        item = _mapping(definition, f"prerequisites.{name}")
        default = item.get("default")
        if not isinstance(default, bool):
            raise ValueError(f"prerequisites.{name}.default must be a boolean")
        prerequisite_defaults[name] = default
        prerequisite_reasons[name] = _string(item.get("reason"), f"prerequisites.{name}.reason")
    trigger_reasons = {
        name: _string(
            _mapping(definition, f"triggers.{name}").get("reason"),
            f"triggers.{name}.reason",
        )
        for name, definition in triggers.items()
    }
    confidence = _bounded(
        _number(risk.get("confidence_level"), "risk_control.confidence_level"),
        "risk_control.confidence_level",
        0.5,
        0.9999,
    )
    maximum_false_clear_rate = _bounded(
        _number(
            risk.get("maximum_false_clear_rate"),
            "risk_control.maximum_false_clear_rate",
        ),
        "risk_control.maximum_false_clear_rate",
        0.0,
        1.0,
    )
    maximum_group_false_clear_rate = _bounded(
        _number(
            risk.get("maximum_group_false_clear_rate"),
            "risk_control.maximum_group_false_clear_rate",
        ),
        "risk_control.maximum_group_false_clear_rate",
        0.0,
        1.0,
    )
    dropout = _bounded(
        _number(model.get("dropout"), "model.dropout"),
        "model.dropout",
        0.0,
        0.9,
    )
    spec = PolicySpec(
        name=_string(raw.get("name"), "name"),
        version=_string(raw.get("version"), "version"),
        description=_string(raw.get("description"), "description"),
        requires_attention_label=_string(
            labels.get("requires_attention"), "labels.requires_attention"
        ),
        no_attention_label=_string(labels.get("no_attention"), "labels.no_attention"),
        prerequisites=tuple(prerequisites),
        prerequisite_defaults=prerequisite_defaults,
        prerequisite_reasons=prerequisite_reasons,
        triggers=tuple(triggers),
        trigger_reasons=trigger_reasons,
        success_reason=_string(clearance.get("success_reason"), "clearance.success_reason"),
        risk_control=RiskControlSpec(
            confidence_level=confidence,
            maximum_false_clear_rate=maximum_false_clear_rate,
            maximum_observed_false_clearances=_integer(
                risk.get("maximum_observed_false_clearances"),
                "risk_control.maximum_observed_false_clearances",
            ),
            maximum_group_false_clear_rate=maximum_group_false_clear_rate,
            minimum_group_attention_examples=_integer(
                risk.get("minimum_group_attention_examples"),
                "risk_control.minimum_group_attention_examples",
            ),
        ),
        model=ModelSpec(
            ensemble_size=_integer(model.get("ensemble_size"), "model.ensemble_size"),
            hidden_size=_integer(model.get("hidden_size"), "model.hidden_size"),
            dropout=dropout,
            epochs=_integer(model.get("epochs"), "model.epochs"),
            batch_size=_integer(model.get("batch_size"), "model.batch_size"),
            learning_rate=_number(model.get("learning_rate"), "model.learning_rate"),
            ranking_margin=_number(model.get("ranking_margin"), "model.ranking_margin"),
            ranking_weight=_number(model.get("ranking_weight"), "model.ranking_weight"),
            consistency_weight=_number(model.get("consistency_weight"), "model.consistency_weight"),
            max_tokens=_integer(model.get("max_tokens"), "model.max_tokens"),
            window_overlap=_integer(model.get("window_overlap"), "model.window_overlap"),
            maximum_windows=_integer(model.get("maximum_windows"), "model.maximum_windows"),
        ),
    )
    if spec.model.ensemble_size < 1:
        raise ValueError("model.ensemble_size must be positive")
    if spec.risk_control.maximum_observed_false_clearances < 0:
        raise ValueError("risk_control.maximum_observed_false_clearances must be non-negative")
    if spec.model.hidden_size < 1 or spec.model.epochs < 1 or spec.model.batch_size < 1:
        raise ValueError("model dimensions, epochs, and batch size must be positive")
    if spec.model.window_overlap >= spec.model.max_tokens:
        raise ValueError("model.window_overlap must be below model.max_tokens")
    return spec
