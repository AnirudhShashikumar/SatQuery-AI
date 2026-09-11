"""Integration coverage for the exported RSVQA Specialist v1 runtime."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import backend
from satquery_agent.models import SpecialistHealth, TaskType
from satquery_agent.router import detect_task
from satquery_agent.specialists.rsvqa_specialist import (
    EXPECTED_CHECKPOINT_SHA256,
    MODEL_USED,
    RSVQASpecialist,
    RSVQASpecialistError,
)


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "rsvqa_specialist_v1"


class FakeSVEManager:
    def __init__(self) -> None:
        self.load_calls = 0
        self.encode_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def health(self) -> SpecialistHealth:
        return SpecialistHealth(status="ready", device="cpu")

    def encode_shared_features(self, image, text, *, content_hash=None):
        self.encode_calls += 1
        assert image.mode == "RGB"
        assert text
        return (
            torch.linspace(-1.0, 1.0, 768).unsqueeze(0),
            torch.linspace(1.0, -1.0, 768).unsqueeze(0),
            {
                "device": "cpu",
                "runtime_ms": 2,
                "model_load_ms": 0,
                "model_reused": True,
                "image_cache_hit": False,
                "text_cache_hit": False,
                "fallback": None,
            },
        )


def _image() -> Image.Image:
    array = np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3) % 255
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    _image().save(buffer, format="PNG")
    return buffer.getvalue()


def _endpoint_prediction(task: str = "presence") -> dict:
    probabilities = [0.12, 0.88] if task != "count" else [0.0] * 201 + [1.0]
    logits = [-1.0, 1.0] if task != "count" else [-10.0] * 201 + [10.0]
    return {
        "answer": "yes" if task != "count" else "201+",
        "confidence": max(probabilities),
        "task": task,
        "logits": logits,
        "probabilities": probabilities,
        "model_used": MODEL_USED,
        "runtime_ms": 3,
        "encoder": {
            "device": "cpu",
            "runtime_ms": 2,
            "model_reused": True,
            "image_cache_hit": True,
            "text_cache_hit": False,
        },
    }


def test_checkpoint_loads_exact_export_without_duplicate_encoder() -> None:
    sve = FakeSVEManager()
    specialist = RSVQASpecialist(sve_manager=sve, model_dir=MODEL_DIR)
    specialist.load()
    assert specialist.loaded
    assert specialist.checkpoint_hash == EXPECTED_CHECKPOINT_SHA256
    assert specialist.health().device == "cpu"
    assert sve.load_calls == 1
    assert sum(parameter.numel() for parameter in specialist._head.parameters()) == 4_297_936


@pytest.mark.parametrize(
    ("question", "task", "classes"),
    [
        ("Is there a river?", "presence", 2),
        ("Are there fewer buildings than roads?", "comp", 2),
        ("Is it a rural or an urban area?", "rural_urban", 2),
        ("How many roads are there?", "count", 202),
    ],
)
def test_prediction_decodes_confidence_and_exported_vocabulary(question: str, task: str, classes: int) -> None:
    sve = FakeSVEManager()
    specialist = RSVQASpecialist(sve_manager=sve, model_dir=MODEL_DIR)
    result = specialist.predict(_image(), question)
    assert result["task"] == task
    assert len(result["logits"]) == classes
    assert len(result["probabilities"]) == classes
    assert sum(result["probabilities"]) == pytest.approx(1.0, abs=1e-6)
    assert result["confidence"] == pytest.approx(max(result["probabilities"]))
    assert result["answer"] in specialist._vocabulary[task]
    assert sve.encode_calls == 1


@pytest.mark.parametrize(
    "question",
    ["Is there a river?", "How many roads?", "Are there fewer buildings than roads?", "Is it rural or urban?"],
)
def test_router_dispatches_exported_families_to_specialist(question: str) -> None:
    task, tools, reason = detect_task(question)
    assert task == TaskType.VQA
    assert tools == ["input_validator", "rsvqa_vqa_specialist"]
    assert "primary prediction source" in reason


def test_image_analysis_endpoint_reports_neural_model_used() -> None:
    client = TestClient(backend.app)
    with patch.object(backend.get_rsvqa_specialist(), "predict", return_value=_endpoint_prediction()):
        response = client.post(
            "/api/analysis/image",
            data={
                "analysis_type": "ground_truth",
                "model_name": "satquery-agent",
                "question": "Is there a river?",
            },
            files={"image": ("scene.png", _png_bytes(), "image/png")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "yes"
    assert body["confidence"] == pytest.approx(0.88)
    assert body["model_used"] == "RSVQA Specialist v1"
    assert body["task"] == "presence"
    assert body["vqa_details"]["probabilities"] == [0.12, 0.88]
    assert [step["tool"] for step in body["execution_trace"]][-1] == "rsvqa_vqa_specialist"


def test_specialist_failure_uses_existing_heuristic_endpoint_path() -> None:
    client = TestClient(backend.app)
    with patch.object(
        backend.get_rsvqa_specialist(),
        "predict",
        side_effect=RSVQASpecialistError("test load failure"),
    ), patch.dict("os.environ", {"SATQUERY_CAPTIONER_ENABLED": "0", "SVE_ENABLED": "0"}):
        response = client.post(
            "/api/analysis/image",
            data={
                "analysis_type": "ground_truth",
                "model_name": "satquery-agent",
                "question": "Is it a rural or an urban area?",
            },
            files={"image": ("scene.png", _png_bytes(), "image/png")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_used"] != "RSVQA Specialist v1"
    assert "rs_vqa" in body["model_used"]
    assert body["answer"] in {"rural", "urban", None}

