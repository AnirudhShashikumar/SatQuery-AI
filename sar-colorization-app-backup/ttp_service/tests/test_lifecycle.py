from __future__ import annotations

import numpy as np

from ttp_service.lifecycle import ModelLifecycle


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
