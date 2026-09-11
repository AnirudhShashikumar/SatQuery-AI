"""Production-contract tests for the TTP-primary bi-temporal path."""

from __future__ import annotations

import io
import os
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend import app
from satquery_agent.models import AgentResponse
from satquery_agent.specialists.ttp_change import TTP_CLIENT, TTPClientResult


def png_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(output, "PNG")
    return output.getvalue()


def learned_result() -> TTPClientResult:
    mask = np.zeros((32, 32), dtype=bool)
    mask[7:25, 8:24] = True
    return TTPClientResult(
        mask=mask,
        changed_percentage=28.125,
        changed_pixels=288,
        region_count=1,
        largest_region_pixels=288,
        runtime_ms=73,
        model_load_ms=1400,
        reused_model=True,
        device="cuda",
        gpu_allocated_mb=812.5,
        gpu_reserved_mb=1024.0,
        gpu_peak_mb=930.25,
        service_trace=[
            {"stage": "probability_map_generation", "status": "success", "decision_threshold": 0.5},
            {"stage": "mask_validation", "status": "success", "changed_pixels": 288},
        ],
    )


def test_agent_health_exposes_complete_ttp_lifecycle() -> None:
    payload = {
        "status": "ready", "lifecycle": "ready", "device": "cuda",
        "checkpoint": "epoch_260.pth", "loaded": True, "load_count": 1,
        "reuse_count": 4, "inference_count": 5, "errors": [],
    }
    with patch.object(TTP_CLIENT, "health_payload", return_value=payload):
        body = TestClient(app).get("/api/agent/health").json()
    assert body["specialists"]["ttp_change_detector"] == payload


def test_real_ttp_contract_is_primary_traceable_and_reportable() -> None:
    before = np.zeros((32, 32, 3), dtype=np.uint8)
    after = before.copy()
    after[8:24, 8:24] = 255
    remote_health = {
        "status": "ready", "lifecycle": "ready", "device": "cuda",
        "checkpoint_verified": True, "model_load_count": 1,
        "model_reuse_count": 3, "inference_count": 4,
    }
    with patch.dict(os.environ, {"TTP_ENABLED": "true", "TTP_DEFAULT_MODE": "hybrid"}), \
         patch.object(TTP_CLIENT, "health", return_value=remote_health), \
         patch.object(TTP_CLIENT, "predict", return_value=learned_result()):
        with TestClient(app) as client:
            response = client.post(
                "/api/agent/query",
                data={
                    "query": "What changed?", "input_mode": "bi_temporal",
                    "primary_modality": "optical", "secondary_modality": "optical",
                    "primary_date": "2024-01-01", "secondary_date": "2025-01-01",
                },
                files={
                    "primary_image": ("before.png", png_bytes(before), "image/png"),
                    "secondary_image": ("after.png", png_bytes(after), "image/png"),
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert set(body) == set(AgentResponse.model_fields)
            assert body["change_engine"] == {
                "mode": "hybrid", "primary_tool": "ttp_change_detector",
                "supporting_tool": "deterministic_change_analyzer",
                "fallback_used": False, "fallback_reason": None,
            }
            change = body["change_analysis"]
            assert change["statistics"]["changed_pixels"] == 288
            assert body["ttp_result"]["checkpoint"] == "epoch_260.pth"
            assert body["ttp_result"]["device"] == "cuda"
            assert change["previews"]["ttp_raw_mask"]

            inference = next(step for step in body["execution"]["steps"] if step["tool"] == "ttp_inference")
            assert inference["duration_ms"] == 73
            assert inference["parameters"]["mask_generated"] is True
            assert inference["parameters"]["gpu_peak_mb"] == 930.25
            assert any(step["tool"] == "ttp_service_probability_map_generation" for step in body["execution"]["steps"])

            report = client.post(
                "/api/agent/report",
                json={"request_id": body["request_id"], "formats": ["json"]},
            )
            assert report.status_code == 200, report.text
            artifact_url = report.json()["artifacts"][0]["url"]
            artifact = client.get(artifact_url)
            assert artifact.status_code == 200
            assert b"epoch_260.pth" in artifact.content
