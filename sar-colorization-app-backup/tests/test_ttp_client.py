from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch

import numpy as np
import requests
from PIL import Image

from satquery_agent.specialists.ttp_change import TTPClientError, TTPServiceClient


def response(status: int, payload: dict | bytes, content_type: str = "application/json") -> requests.Response:
    value = requests.Response()
    value.status_code = status
    value.headers["content-type"] = content_type
    value._content = json.dumps(payload).encode() if isinstance(payload, dict) else payload
    return value


def mask_png(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(values.astype(np.uint8)).save(output, "PNG")
    return output.getvalue()


HEALTH = {
    "status": "ready", "service": "ttp_change_detector", "model": "TTP", "training_dataset": "LEVIR-CD",
    "checkpoint_fingerprint": "60294429b3d", "checkpoint_verified": True, "device": "cuda", "lifecycle": "ready",
}


class FakeSession:
    def __init__(self, prediction: requests.Response, artifact: requests.Response | None = None) -> None:
        self.prediction = prediction
        self.artifact = artifact
        self.health_calls = 0

    def get(self, url, **kwargs):
        if url.endswith("/health"):
            self.health_calls += 1
            return response(200, HEALTH)
        return self.artifact or response(404, {})

    def post(self, url, **kwargs):
        return self.prediction


class TTPClientTests(unittest.TestCase):
    def payload(self, **updates):
        body = {
            "status": "success", "model": "TTP", "training_dataset": "LEVIR-CD", "checkpoint": "epoch_260.pth",
            "checkpoint_fingerprint": "60294429b3d", "device": "cuda", "input_width": 8, "input_height": 8,
            "changed_pixels": 16, "changed_percentage": 25.0, "region_count": 1, "largest_region_pixels": 16,
            "runtime": {"inference_ms": 50, "model_load_ms": 1000}, "reused_model": True,
            "warnings": [], "limitations": [], "trace": [],
            "artifacts": [{"kind": "raw_binary_mask", "endpoint": "/artifacts/" + "a" * 32}],
        }
        body.update(updates)
        return body

    def test_success_validates_mask_and_reuses_cached_health(self):
        values = np.zeros((8, 8), dtype=np.uint8)
        values[2:6, 2:6] = 255
        session = FakeSession(response(200, self.payload()), response(200, mask_png(values), "image/png"))
        client = TTPServiceClient(session)
        with patch.dict(os.environ, {"TTP_ENABLED": "true", "TTP_SERVICE_URL": "http://service.internal"}):
            first = client.predict(b"before", b"after", "before.png", "after.png", 8, 8, "request-1")
            client.health()
        self.assertEqual(first.changed_pixels, 16)
        self.assertTrue(first.reused_model)
        self.assertEqual(session.health_calls, 1)

    def test_cuda_oom_is_typed_and_safe(self):
        session = FakeSession(response(503, {"detail": {"code": "CUDA_OOM", "message": "internal details ignored"}}))
        client = TTPServiceClient(session)
        with patch.dict(os.environ, {"TTP_ENABLED": "true", "TTP_SERVICE_URL": "http://service.internal"}):
            with self.assertRaises(TTPClientError) as caught:
                client.predict(b"before", b"after", "before.png", "after.png", 8, 8, "request-2")
        self.assertEqual(caught.exception.code, "CUDA_OOM")
        self.assertNotIn("internal", caught.exception.public_message)

    def test_non_binary_mask_is_rejected(self):
        values = np.zeros((8, 8), dtype=np.uint8)
        values[2:6, 2:6] = 127
        session = FakeSession(response(200, self.payload()), response(200, mask_png(values), "image/png"))
        client = TTPServiceClient(session)
        with patch.dict(os.environ, {"TTP_ENABLED": "true", "TTP_SERVICE_URL": "http://service.internal"}):
            with self.assertRaises(TTPClientError) as caught:
                client.predict(b"before", b"after", "before.png", "after.png", 8, 8, "request-3")
        self.assertEqual(caught.exception.code, "INVALID_MASK")

    def test_provenance_mismatch_is_rejected(self):
        session = FakeSession(response(200, self.payload(checkpoint_fingerprint="wrong")))
        client = TTPServiceClient(session)
        with patch.dict(os.environ, {"TTP_ENABLED": "true", "TTP_SERVICE_URL": "http://service.internal"}):
            with self.assertRaises(TTPClientError) as caught:
                client.predict(b"before", b"after", "before.png", "after.png", 8, 8, "request-4")
        self.assertEqual(caught.exception.code, "MALFORMED_RESPONSE")


if __name__ == "__main__":
    unittest.main()
