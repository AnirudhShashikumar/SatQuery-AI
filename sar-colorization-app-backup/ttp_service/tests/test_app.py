from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from ttp_service.app import create_app


class FakeLifecycle:
    state = "ready"
    checkpoint_verified = True
    model_load_ms = 1200
    model_load_count = 1
    inference_count = 1
    model_reuse_count = 1

    def load(self) -> None:
        self.state = "ready"

    def predict(self, earlier: bytes, later: bytes, earlier_suffix: str, later_suffix: str):
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 5:13] = True
        return mask, True, 42, {"gpu_allocated_mb": 900.0, "gpu_reserved_mb": 1024.0, "gpu_peak_mb": 1100.0}


def png(value: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), (value, value, value)).save(output, "PNG")
    return output.getvalue()


def test_health_and_prediction_use_opaque_artifacts() -> None:
    with TestClient(create_app(FakeLifecycle())) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ready"
        response = client.post(
            "/predict",
            files={
                "earlier_image": ("earlier.png", png(20), "image/png"),
                "later_image": ("later.png", png(80), "image/png"),
            },
            data={"request_id": "safe-request"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["changed_pixels"] == 64
        assert payload["reused_model"] is True
        raw = next(item for item in payload["artifacts"] if item["kind"] == "raw_binary_mask")
        assert "/" not in raw["id"]
        artifact = client.get(raw["endpoint"])
        assert artifact.status_code == 200
        assert artifact.headers["content-type"] == "image/png"


def test_dimension_mismatch_is_rejected_without_inference() -> None:
    output = io.BytesIO()
    Image.new("RGB", (15, 16)).save(output, "PNG")
    with TestClient(create_app(FakeLifecycle())) as client:
        response = client.post(
            "/predict",
            files={
                "earlier_image": ("earlier.png", png(20), "image/png"),
                "later_image": ("later.png", output.getvalue(), "image/png"),
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "DIMENSION_MISMATCH"
