import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from legal_decision_model.constants import NO_ATTENTION
from legal_decision_model.policy_model import (
    FINAL_TASK,
    IN_SCOPE_TASK,
    LayaFeatureEncoder,
    PolicyHead,
    PolicyHeadConfig,
    build_feature_cache,
    load_head,
    safe_score,
    save_head,
    task_names,
    task_targets,
)
from legal_decision_model.policy_schema import load_policy
from legal_decision_model.records import DecisionRecord


class FakeEncoder:
    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        assert batch_size == 2
        return torch.tensor([[float(len(text)), 1.0] for text in texts])


class ConcurrentTokenizer:
    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._active = 0
        self.maximum_concurrency = 0

    def __call__(
        self,
        text: str | list[str],
        *,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int] | torch.Tensor]:
        del padding, add_special_tokens
        with self._state_lock:
            self._active += 1
            self.maximum_concurrency = max(self.maximum_concurrency, self._active)
        try:
            time.sleep(0.01)
            if isinstance(text, str):
                token_count = len(text.split())
                if self.maximum_concurrency > 1 and max_length is not None:
                    token_count = min(token_count, max_length)
                return {"input_ids": list(range(token_count))}
            width = min(max(len(value.split()) for value in text), max_length or 10_000)
            input_ids = torch.ones((len(text), width), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            assert return_tensors == "pt"
            assert truncation
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        finally:
            with self._state_lock:
                self._active -= 1


class ConcurrentBackbone:
    config = SimpleNamespace(hidden_size=2)

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(
            last_hidden_state=torch.ones((*input_ids.shape, 2), dtype=torch.float32)
        )


def _record() -> DecisionRecord:
    policy = load_policy()
    facts = policy.default_facts()
    return DecisionRecord(
        record_id="record",
        pair_id="pair",
        split="train",
        family="contracts",
        style="ticket",
        template_id="template",
        text="synthetic request",
        facts=facts,
        label=NO_ATTENTION,
        reasons=(policy.success_reason,),
        scenario_id="scenario",
        rendering_id="a",
    )


def test_task_targets_follow_policy_order() -> None:
    policy = load_policy()
    record = _record()
    names = task_names(policy)
    targets = task_targets(record, policy)
    assert names == (*policy.prerequisites, *policy.triggers, IN_SCOPE_TASK, FINAL_TASK)
    assert len(targets) == len(names)
    assert targets[names.index(policy.prerequisites[0])] == 1.0
    assert all(targets[names.index(name)] == 0.0 for name in policy.triggers)
    assert targets[names.index(IN_SCOPE_TASK)] == 1.0
    assert targets[names.index(FINAL_TASK)] == 0.0


def test_safe_score_is_monotonic_in_risk() -> None:
    policy = load_policy()
    probabilities = {
        **dict.fromkeys(policy.prerequisites, 0.95),
        **dict.fromkeys(policy.triggers, 0.05),
        IN_SCOPE_TASK: 0.95,
        FINAL_TASK: 0.05,
    }
    baseline = safe_score(probabilities, policy)
    probabilities[policy.triggers[0]] = 0.9
    assert safe_score(probabilities, policy) < baseline


def test_policy_head_shape_and_serialization(tmp_path: Path) -> None:
    config = PolicyHeadConfig(
        input_size=4,
        hidden_size=3,
        task_names=("a", "b"),
        families=("contracts", "privacy"),
        dropout=0.0,
    )
    head = PolicyHead(config)
    head.eval()
    features = torch.randn(5, 4)
    expected_tasks, expected_families = head(features)
    path = tmp_path / "head.safetensors"
    save_head(path, head)
    loaded = load_head(path, config, torch.device("cpu"))
    tasks, families = loaded(features)
    assert tasks.shape == (5, 2)
    assert families.shape == (5, 2)
    assert torch.equal(tasks, expected_tasks)
    assert torch.equal(families, expected_families)


def test_non_finite_head_is_rejected(tmp_path: Path) -> None:
    config = PolicyHeadConfig(
        input_size=4,
        hidden_size=3,
        task_names=("a", "b"),
        families=("contracts", "privacy"),
        dropout=0.0,
    )
    head = PolicyHead(config)
    path = tmp_path / "head.safetensors"
    save_head(path, head)
    state = load_file(str(path))
    first = next(iter(state))
    state[first].flatten()[0] = float("nan")
    save_file(state, str(path))
    with pytest.raises(ValueError, match="non-finite"):
        load_head(path, config, torch.device("cpu"))


def test_feature_cache_preserves_record_alignment() -> None:
    policy = load_policy()
    first = _record()
    second = DecisionRecord(
        **{
            **first.to_dict(),
            "record_id": "second",
            "text": "longer synthetic request",
        }
    )
    cache = build_feature_cache(
        [first, second],
        policy,
        FakeEncoder(),  # type: ignore[arg-type]
        ["contracts"],
        batch_size=2,
    )
    assert cache.records == (first, second)
    assert cache.features[:, 0].tolist() == [17.0, 24.0]
    assert cache.task_targets.shape == (2, len(task_names(policy)))
    assert cache.family_targets.tolist() == [0, 0]


def test_concurrent_encoding_does_not_truncate_window_planning() -> None:
    tokenizer = ConcurrentTokenizer()
    encoder = LayaFeatureEncoder.__new__(LayaFeatureEncoder)
    encoder.tokenizer = tokenizer
    encoder.encoder = ConcurrentBackbone()
    encoder.device = torch.device("cpu")
    encoder.max_tokens = 512
    long_text = " ".join(f"token-{index}" for index in range(1_951))

    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in range(20):
            feature_future = executor.submit(encoder.encode, ["short request"], 1)
            token_future = executor.submit(encoder.token_ids, long_text)
            assert len(token_future.result()) == 1_951
            assert feature_future.result().shape == (1, 2)

    assert tokenizer.maximum_concurrency == 1
