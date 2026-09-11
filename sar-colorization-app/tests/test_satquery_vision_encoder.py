from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from PIL import Image

from satquery_agent.satquery_vision_encoder import SatQueryVisionEncoder
from satquery_agent.sve_artifacts import SVELoadFailure, verify_sve_artifacts

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "satquery_vision_encoder_v1"


class FakeModel:
    def __init__(self, state: OrderedDict[str, torch.Tensor]) -> None:
        self._state = state
        self.loaded: dict[str, torch.Tensor] = {}

    def state_dict(self):
        return self._state

    def load_state_dict(self, values, strict=False):
        self.loaded = dict(values)
        return SimpleNamespace(
            missing_keys=[key for key in self._state if key not in values],
            unexpected_keys=[key for key in values if key not in self._state],
        )

    def eval(self): return self
    def parameters(self): return []
    def to(self, *args, **kwargs): return self
    def encode_image(self, batch): return torch.arange(batch.shape[0] * 768, dtype=torch.float32).reshape(batch.shape[0], 768) + 1
    def encode_text(self, tokens): return torch.arange(tokens.shape[0] * 768, dtype=torch.float32).reshape(tokens.shape[0], 768) + 1


def test_exact_adapter_keys_and_normalized_embeddings() -> None:
    artifacts = verify_sve_artifacts(MODEL_DIR)
    payload = torch.load(artifacts.adapter_path, map_location="cpu", weights_only=True)
    state = OrderedDict((key.removeprefix("clip_model."), torch.zeros_like(value)) for key, value in payload["adapter_state_dict"].items())
    state["visual.class_embedding"] = torch.zeros(1024)
    model = FakeModel(state)

    def factory(*args, **kwargs):
        return model, None, lambda image: torch.zeros(3, 224, 224)

    encoder = SatQueryVisionEncoder(
        artifacts,
        "cpu",
        model_factory=factory,
        tokenizer_factory=lambda _: lambda texts: torch.ones(len(texts), 77, dtype=torch.long),
    )
    encoder.load()
    assert len(model.loaded) == 28
    image_embedding = encoder.encode_image(Image.new("RGB", (32, 32), "green"))
    text_embedding = encoder.encode_text(["forest", "water"])
    assert image_embedding.shape == (1, 768)
    assert text_embedding.shape == (2, 768)
    assert torch.allclose(image_embedding.norm(dim=-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(text_embedding.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.isfinite(encoder.cosine_similarity(image_embedding, text_embedding)).all()


def test_missing_adapter_key_is_rejected(monkeypatch) -> None:
    artifacts = verify_sve_artifacts(MODEL_DIR)
    real_load = torch.load
    payload = real_load(artifacts.adapter_path, map_location="cpu", weights_only=True)
    state = OrderedDict((key.removeprefix("clip_model."), torch.zeros_like(value)) for key, value in payload["adapter_state_dict"].items())
    state["visual.class_embedding"] = torch.zeros(1024)
    model = FakeModel(state)
    damaged = {**payload, "adapter_state_dict": dict(payload["adapter_state_dict"])}
    damaged["adapter_state_dict"].pop(next(iter(damaged["adapter_state_dict"])))
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: damaged)
    encoder = SatQueryVisionEncoder(
        artifacts, "cpu",
        model_factory=lambda *args, **kwargs: (model, None, lambda image: torch.zeros(3, 224, 224)),
        tokenizer_factory=lambda _: lambda texts: torch.ones(len(texts), 77, dtype=torch.long),
    )
    with pytest.raises(SVELoadFailure):
        encoder.load()
