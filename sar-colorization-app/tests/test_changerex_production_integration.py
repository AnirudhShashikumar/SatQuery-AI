"""Focused production integration tests for ChangerEx-primary change analysis."""

from __future__ import annotations

import io
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend import app
from satquery_agent.models import AgentResponse
from satquery_agent.specialists.change_analysis import statistics_from_binary_mask
from satquery_agent.specialists.changerex_change import (
    CHANGEREX_ENGINE,
    ChangerExClientResult,
    ChangerExProductionError,
    selected_change_engine,
)
from satquery_agent.specialists import changerex_change as production_adapter


def _png(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(values.astype(np.uint8)).save(output, "PNG")
    return output.getvalue()


def _learned_result(*, reused: bool = True) -> ChangerExClientResult:
    mask = np.zeros((32, 32), dtype=bool)
    mask[7:25, 8:24] = True
    return ChangerExClientResult(
        mask=mask,
        changed_percentage=28.125,
        changed_pixels=288,
        region_count=1,
        largest_region_pixels=288,
        runtime_ms=93,
        model_load_ms=812,
        reused_model=reused,
        device="mps",
        threshold=0.5,
        stage_durations_ms={
            "inference": 71,
            "probability_extraction": 4,
            "thresholding": 2,
            "connected_components": 3,
        },
    )


def _submit(client: TestClient):
    before = np.zeros((32, 32, 3), dtype=np.uint8)
    after = before.copy()
    after[8:24, 8:24] = 255
    return client.post(
        "/api/agent/query",
        data={
            "query": "What changed?",
            "input_mode": "bi_temporal",
            "primary_modality": "optical",
            "secondary_modality": "optical",
            "primary_date": "2024-01-01",
            "secondary_date": "2025-01-01",
        },
        files={
            "primary_image": ("before.png", _png(before), "image/png"),
            "secondary_image": ("after.png", _png(after), "image/png"),
        },
    )


def test_changerex_is_default_and_ttp_remains_explicit_alternate(monkeypatch) -> None:
    monkeypatch.setenv("SATQUERY_CHANGEREX_ENABLED", "true")
    monkeypatch.delenv("SATQUERY_CHANGE_ENGINE", raising=False)
    monkeypatch.delenv("TTP_DEFAULT_MODE", raising=False)
    assert selected_change_engine() == "changerex"
    monkeypatch.setenv("SATQUERY_CHANGE_ENGINE", "ttp")
    assert selected_change_engine() == "ttp"


def test_health_exposes_changerex_lifecycle() -> None:
    payload = {
        "status": "ready", "lifecycle": "ready", "model": "ChangerEx",
        "device": "mps", "checkpoint": "official.pth", "loaded": True,
        "load_count": 1, "reuse_count": 2, "inference_count": 3, "errors": [],
    }
    with patch.object(CHANGEREX_ENGINE, "health_payload", return_value=payload):
        response = TestClient(app).get("/api/agent/health")
    assert response.status_code == 200
    assert response.json()["specialists"]["changerex_change_detector"] == payload


def test_production_adapter_configures_the_standalone_singleton_once(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "official.pth"
    checkpoint.write_bytes(b"test placeholder; strict loading is mocked here")
    monkeypatch.setenv("SATQUERY_CHANGEREX_ENABLED", "true")
    monkeypatch.setenv("SATQUERY_CHANGEREX_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SATQUERY_CHANGEREX_DEVICE", "cpu")
    raw = SimpleNamespace(
        binary_mask=np.zeros((8, 8), dtype=np.uint8),
        changed_percentage=0.0,
        changed_pixel_count=0,
        connected_component_count=0,
        largest_component_size=0,
        runtime=SimpleNamespace(inference_seconds=0.01, load_seconds=0.02),
        load_reuse_status={"was_reused": False},
        selected_device="cpu",
        threshold=0.5,
        stage_durations_ms={},
        warnings=[],
        limitations=[],
    )
    engine = production_adapter.ChangerExChangeEngine()
    image = Image.new("RGB", (8, 8))
    with patch.object(production_adapter, "configure_lifecycle") as configure, patch.object(
        production_adapter, "predict_change", return_value=raw
    ):
        engine.predict(image, image)
        raw.load_reuse_status = {"was_reused": True}
        second = engine.predict(image, image)
    image.close()
    configure.assert_called_once()
    assert second.reused_model is True


def test_changerex_mask_is_primary_with_unchanged_response_schema_and_full_trace() -> None:
    environment = {
        "SATQUERY_CHANGE_ENGINE": "changerex",
        "SATQUERY_CHANGEREX_ENABLED": "true",
    }
    with patch.dict(os.environ, environment), patch.object(
        CHANGEREX_ENGINE, "predict", return_value=_learned_result()
    ):
        with TestClient(app) as client:
            response = _submit(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == set(AgentResponse.model_fields)
    assert body["change_engine"] == {
        "mode": "hybrid",
        "primary_tool": "changerex_change_detector",
        "supporting_tool": "deterministic_change_analyzer",
        "fallback_used": False,
        "fallback_reason": None,
    }
    assert body["change_analysis"]["statistics"]["changed_pixels"] == 288
    assert body["ttp_result"]["model"].startswith("ChangerEx")
    assert body["change_analysis"]["previews"]["ttp_raw_mask"]
    assert "learned evidence identifies" in body["answer"].lower()
    assert "changed pixels" not in body["answer"].lower()
    assert body["semantic_change_summary"]["generated_by"] == "local_semantic_interpreter"
    tools = {step["tool"] for step in body["execution"]["steps"]}
    assert {
        "changerex_model_reuse",
        "changerex_inference",
        "changerex_probability_extraction",
        "changerex_thresholding",
        "changerex_connected_components",
        "changerex_overlay_generation",
        "semantic_change_interpretation",
    }.issubset(tools)


def test_changerex_failure_preserves_deterministic_result_and_surfaces_warning() -> None:
    environment = {
        "SATQUERY_CHANGE_ENGINE": "changerex",
        "SATQUERY_CHANGEREX_ENABLED": "true",
    }
    failure = ChangerExProductionError(
        "CHECKPOINT_UNAVAILABLE",
        "ChangerEx checkpoint is unavailable; the deterministic change analyzer produced the result.",
    )
    with patch.dict(os.environ, environment), patch.object(
        CHANGEREX_ENGINE, "predict", side_effect=failure
    ):
        with TestClient(app) as client:
            body = _submit(client).json()
    assert body["change_engine"]["mode"] == "deterministic_fallback"
    assert body["change_engine"]["fallback_used"] is True
    assert body["change_analysis"]["statistics"] == body["change_analysis"]["deterministic_statistics"]
    assert any("checkpoint is unavailable" in warning for warning in body["warnings"])
    tools = {step["tool"] for step in body["execution"]["steps"]}
    assert "changerex_model_load" in tools
    assert "deterministic_fallback" in tools


def test_region_statistics_accept_the_shared_reduced_analysis_grid() -> None:
    mask = np.zeros((512, 512), dtype=bool)
    mask[10:20, 20:40] = True
    statistics = statistics_from_binary_mask(mask, source_width=2048, source_height=2048)
    assert statistics.analysis_width == 512
    assert statistics.source_width == 2048
    assert statistics.changed_pixels == 200
