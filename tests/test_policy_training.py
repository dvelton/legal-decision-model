from pathlib import Path

import pytest
import torch

from legal_decision_model.policy_training import _resumable_encode


class InterruptingEncoder:
    output_size = 2
    device = torch.device("cpu")

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        assert batch_size == 2
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated interruption")
        return torch.tensor([[float(len(text)), 1.0] for text in texts])


def test_feature_encoding_resumes_completed_chunks(tmp_path: Path) -> None:
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    cache_path = tmp_path / "features.safetensors"
    interrupted = InterruptingEncoder(fail_on_call=2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _resumable_encode(
            texts,
            interrupted,  # type: ignore[arg-type]
            cache_path,
            batch_size=2,
            chunk_size=2,
        )

    resumed = InterruptingEncoder()
    features, chunk_dir = _resumable_encode(
        texts,
        resumed,  # type: ignore[arg-type]
        cache_path,
        batch_size=2,
        chunk_size=2,
    )

    assert resumed.calls == 2
    assert features[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert chunk_dir.is_dir()
