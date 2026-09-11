from __future__ import annotations

import numpy as np
import pytest

from ttp_service.lifecycle import LifecycleError, ModelLifecycle


class FakeEngine:
    loads = 0

    def load(self) -> None:
        self.loads += 1

    def predict(self, *args):
        return np.zeros((2, 2), dtype=bool)


class VerifiedLifecycle(ModelLifecycle):
    @property
    def checkpoint_verified(self) -> bool:
        return True


def test_model_loads_once_and_is_reused(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    engine = FakeEngine()
    lifecycle = VerifiedLifecycle(lambda: engine)
    lifecycle.load()
    lifecycle.load()
    assert engine.loads == 1
    _, first_reuse, _, _ = lifecycle.predict(b"a", b"b", ".png", ".png")
    _, second_reuse, _, _ = lifecycle.predict(b"a", b"b", ".png", ".png")
    assert first_reuse is False
    assert second_reuse is True
    assert lifecycle.model_reuse_count == 1


def test_first_prediction_lazy_loads_exactly_once(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    engine = FakeEngine()
    lifecycle = VerifiedLifecycle(lambda: engine)
    _, first_reuse, _, _ = lifecycle.predict(b"a", b"b", ".png", ".png")
    _, second_reuse, _, _ = lifecycle.predict(b"a", b"b", ".png", ".png")
    assert lifecycle.state == "ready"
    assert lifecycle.model_load_count == 1
    assert engine.loads == 1
    assert first_reuse is False
    assert second_reuse is True
    assert lifecycle.health()["checkpoint"] == "epoch_260.pth"
    assert lifecycle.health()["reuse_count"] == 1


def test_checkpoint_load_failure_is_sticky() -> None:
    class FailingEngine:
        attempts = 0

        def load(self) -> None:
            self.attempts += 1
            raise RuntimeError("broken checkpoint")

    engine = FailingEngine()
    lifecycle = VerifiedLifecycle(lambda: engine)  # type: ignore[arg-type]
    with pytest.raises(LifecycleError):
        lifecycle.predict(b"a", b"b", ".png", ".png")
    with pytest.raises(LifecycleError):
        lifecycle.predict(b"a", b"b", ".png", ".png")
    assert engine.attempts == 1
    assert lifecycle.state == "failed"
    assert lifecycle.health()["loaded"] is False
    assert lifecycle.health()["errors"] == ["MODEL_LOAD_FAILED"]
