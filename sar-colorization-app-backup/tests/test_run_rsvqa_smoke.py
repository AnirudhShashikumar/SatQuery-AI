"""Focused tests for the resumable RSVQA smoke benchmark runner."""

from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import requests
from rasterio.transform import from_origin

from scripts import run_rsvqa_smoke as smoke


@pytest.mark.parametrize(
    ("value", "question_type", "expected"),
    [
        (" YES! ", "presence", "yes"),
        ("negative", "comp", "no"),
        ("an urban area", "rural_urban", "urban"),
        ("Rural.", "rural_urban", "rural"),
        (0, "count", "0"),
        (246, "count", "201+"),
        ("201+", "count", "201+"),
        ("002", "count", None),
        ("There are 2 roads", "count", None),
        (-1, "count", None),
        (None, "presence", None),
    ],
)
def test_answer_normalization(value: object, question_type: str, expected: str) -> None:
    assert smoke.normalize_answer(value, question_type) == expected


def test_image_path_resolution_ignores_manifest_colab_path(tmp_path: Path) -> None:
    record = {"image_id": 232, "image_path": "/content/drive/wrong/location.tif"}
    assert smoke.resolve_image_path(record, tmp_path) == tmp_path.resolve() / "232.tif"


def _write_tiff(path: Path) -> None:
    rows = cols = 8
    base = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
    data = np.stack((base, base * 2 + 10, np.full_like(base, 7)))
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=cols,
        height=rows,
        count=3,
        dtype=data.dtype,
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(data)


def test_tiff_to_png_conversion_is_cached(tmp_path: Path) -> None:
    source = tmp_path / "232.tif"
    converted = tmp_path / "converted"
    _write_tiff(source)
    first_path, first_hit = smoke.convert_tiff_to_png(source, converted, 232)
    first_mtime = first_path.stat().st_mtime_ns
    assert first_hit is False
    assert first_path.name == "232.png"
    with rasterio.open(first_path) as image:
        assert image.count == 3
        assert (image.width, image.height) == (8, 8)
    time.sleep(0.001)
    with patch.object(smoke.rasterio, "open", side_effect=AssertionError("cache miss")):
        second_path, second_hit = smoke.convert_tiff_to_png(source, converted, 232)
    assert second_path == first_path
    assert second_hit is True
    assert second_path.stat().st_mtime_ns == first_mtime


def test_metric_aggregation_counts_errors_nulls_and_latency() -> None:
    predictions = [
        {"question_id": "1", "image_id": "1", "question_type": "presence", "completed": True, "correct": True, "normalized_prediction": "yes", "confidence": 0.8, "effective_latency_ms": 100, "reuse_observations": {}},
        {"question_id": "2", "image_id": "2", "question_type": "presence", "completed": True, "correct": False, "normalized_prediction": None, "confidence": 0.2, "effective_latency_ms": 200, "reuse_observations": {}},
        {"question_id": "3", "image_id": "3", "question_type": "count", "completed": True, "correct": False, "normalized_prediction": "0", "confidence": 0.5, "effective_latency_ms": 300, "reuse_observations": {}},
        {"question_id": "4", "image_id": "4", "question_type": "count", "completed": False, "correct": False, "normalized_prediction": None, "confidence": None, "effective_latency_ms": 400, "reuse_observations": {}},
    ]
    summary = smoke.aggregate_metrics(predictions)
    assert summary["completed_count"] == 3
    assert summary["endpoint_error_count"] == 1
    assert summary["null_answer_count"] == 1
    assert summary["overall_accuracy"] == pytest.approx(0.25)
    assert summary["accuracy_by_question_type"]["presence"]["accuracy"] == pytest.approx(0.5)
    assert summary["average_confidence"] == pytest.approx(0.5)
    assert summary["average_latency_ms"] == pytest.approx(200.0)
    assert summary["median_latency_ms"] == pytest.approx(200.0)
    assert summary["p95_latency_ms"] == pytest.approx(290.0)


class _FakeResponse:
    status_code = 200
    ok = True

    @staticmethod
    def json() -> dict:
        return {
            "answer": "yes",
            "confidence": 0.75,
            "task": "presence_vqa",
            "status": "success",
            "result_status": "COMPLETED",
            "processing_time_ms": 84,
            "model_used": "rs_vqa",
            "warnings": [],
        }


class _FailThenSucceedSession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        self.calls += 1
        if self.calls == 1:
            raise requests.ConnectionError("temporary endpoint failure")
        return _FakeResponse()


def test_endpoint_error_does_not_stop_following_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    png_path = tmp_path / "cached.png"
    png_path.write_bytes(b"png")
    monkeypatch.setattr(smoke, "convert_tiff_to_png", lambda *_args: (png_path, True))
    records = [
        {"question_id": 1, "image_id": 10, "image_path": "/ignored", "question": "Is there water?", "question_type": "presence", "ground_truth": "yes"},
        {"question_id": 2, "image_id": 11, "image_path": "/ignored", "question": "Is there water?", "question_type": "presence", "ground_truth": "yes"},
    ]
    session = _FailThenSucceedSession()
    config = smoke.BenchmarkConfig(tmp_path, "http://example.test/api/analysis/image", tmp_path / "output")
    rows = smoke.run_records(records, config, session=session, progress_stream=io.StringIO())
    assert session.calls == 2
    assert rows[0]["completed"] is False
    assert rows[0]["error_stage"] == "endpoint_request"
    assert rows[1]["completed"] is True
    assert rows[1]["correct"] is True
    assert smoke.aggregate_metrics(rows)["endpoint_error_count"] == 1
