from __future__ import annotations

import hashlib

import torch
import pytest
from PIL import Image

from satquery_agent.services.sve_service import SVEManager


class MockEncoder:
    def __init__(self, artifacts, device): self.device = device
    def load(self): pass
    def close(self): pass
    def move_to(self, device): self.device = device
    def encode_image(self, image):
        value = torch.linspace(1, 2, 768).unsqueeze(0)
        return value / value.norm(dim=-1, keepdim=True)
    def encode_text(self, texts):
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            generator = torch.Generator().manual_seed(seed)
            value = torch.rand(768, generator=generator)
            rows.append(value / value.norm())
        return torch.stack(rows)
    @staticmethod
    def cosine_similarity(first, second): return (first @ second.T).clamp(-1, 1)


class MPSFallbackEncoder(MockEncoder):
    def encode_image(self, image):
        if self.device == "mps":
            from satquery_agent.sve_artifacts import SVEInferenceFailure
            raise SVEInferenceFailure("unsupported mps operation")
        return super().encode_image(image)


def test_lazy_load_reuse_scene_priors_caption_and_vqa(monkeypatch) -> None:
    monkeypatch.setenv("SVE_ENABLED", "true")
    monkeypatch.setenv("SVE_DEVICE", "cpu")
    service = SVEManager(encoder_factory=MockEncoder)
    image = Image.new("RGB", (64, 64), "green")
    first = service.analyze(image, "a" * 64, captions=["forest and fields"], vqa_category="presence_water", vqa_answer="No water is visible.")
    second = service.analyze(image, "a" * 64, captions=["forest and fields"])
    assert first.result.available
    assert first.result.scene_priors
    assert first.result.caption_consistency is not None
    assert first.result.vqa_consistency is not None
    assert second.result.available
    metrics = service.metrics()
    assert metrics["model_load_count"] == 1
    assert metrics["model_reuse_count"] == 1
    assert metrics["inference_count"] == 2
    assert metrics["cache_hit_rate_percent"] is not None


def test_raw_sar_is_explicitly_skipped(monkeypatch) -> None:
    monkeypatch.setenv("SVE_ENABLED", "true")
    service = SVEManager(encoder_factory=MockEncoder)
    result = service.unsupported_raw_sar_result()
    assert result.status == "unsupported_input"
    assert result.semantic_comparison.status == "skipped_raw_sar"
    assert "Raw SAR" in result.semantic_comparison.disclaimer


def test_retrieval_returns_only_opaque_ids_and_scores(monkeypatch) -> None:
    monkeypatch.setenv("SVE_ENABLED", "true")
    monkeypatch.setenv("SVE_DEVICE", "cpu")
    service = SVEManager(encoder_factory=MockEncoder)
    image = Image.new("RGB", (32, 32), "blue")
    matches = service.retrieve("water", [("evidence_1", image, "c" * 64), ("evidence_2", image, "d" * 64)])
    assert set(matches[0]) == {"evidence_id", "similarity"}
    assert len(matches) == 2


def test_disabled_fallback_does_not_raise(monkeypatch) -> None:
    monkeypatch.setenv("SVE_ENABLED", "false")
    service = SVEManager(encoder_factory=MockEncoder)
    call = service.analyze(Image.new("RGB", (32, 32)), "b" * 64)
    assert not call.result.available
    assert call.result.status == "disabled"
    assert call.trace[-1]["tool"] == "sve_fallback"


def test_device_priority_and_explicit_selection(monkeypatch) -> None:
    monkeypatch.setenv("SVE_ENABLED", "true")
    monkeypatch.setenv("SVE_DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    service = SVEManager(encoder_factory=MockEncoder)
    assert service._select_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert service._select_device() == "mps"
    monkeypatch.setenv("SVE_DEVICE", "cpu")
    assert service._select_device() == "cpu"


def test_mps_failure_is_traced_and_retried_on_cpu(monkeypatch) -> None:
    monkeypatch.setenv("SVE_ENABLED", "true")
    monkeypatch.setenv("SVE_DEVICE", "mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    service = SVEManager(encoder_factory=MPSFallbackEncoder)
    call = service.analyze(Image.new("RGB", (32, 32), "green"), "e" * 64)
    assert call.result.available
    assert call.result.device == "cpu"
    assert "retried on CPU" in call.result.fallback
    assert service.metrics()["mps_fallback_count"] == 1


def test_checksum_failure_is_publicly_safe_and_sticky(monkeypatch) -> None:
    from satquery_agent.sve_artifacts import SVEChecksumMismatch

    monkeypatch.setenv("SVE_ENABLED", "true")
    monkeypatch.setattr(
        "satquery_agent.services.sve_service.verify_sve_artifacts",
        lambda: (_ for _ in ()).throw(SVEChecksumMismatch("/private/unsafe/checkpoint path")),
    )
    service = SVEManager(encoder_factory=MockEncoder)
    call = service.analyze(Image.new("RGB", (32, 32)), "f" * 64)
    assert call.result.status == "checksum_failure"
    assert "/private/" not in call.result.warning
    assert service.state == "failed"
    with pytest.raises(Exception):
        service.load()
    assert service.metrics()["checksum_failure_count"] == 1
