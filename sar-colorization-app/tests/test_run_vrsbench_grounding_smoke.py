"""Focused tests for the VRSBench Grounding Smoke-100 evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_vrsbench_grounding_smoke as bench


def row(index: int, *, clipped: bool = False, boxes: int = 1, iou: float = 0.6) -> dict:
    return {
        "manifest_index": index, "smoke_id": str(index), "image": f"{index}.png", "image_path": f"{index}.png",
        "object_class": "bridge", "referring_sentence": "The bridge.", "ground_truth_bbox": [0.1, 0.1, 0.5, 0.5],
        "bbox_was_clipped": clipped, "object_position": "center", "relative_position": "",
        "object_size": "small", "relative_size": "", "is_unique": True, "completed": True, "error": None,
        "latency_ms": 10.0 + index, "predicted_box_count": boxes,
        "predicted_boxes": [] if not boxes else [{"bbox_normalized": [0.1, 0.1, 0.5, 0.5], "bbox_pixels": [10, 10, 50, 50], "score": 0.8, "label": "bridge"}],
        "best_prediction_index": 0 if boxes else None, "best_prediction_bbox": [0.1, 0.1, 0.5, 0.5] if boxes else None,
        "best_prediction_label": "bridge" if boxes else None, "best_prediction_score": 0.8 if boxes else None,
        "best_iou": iou if boxes else 0.0, "highest_confidence_index": 0 if boxes else None,
        "highest_confidence_score": 0.8 if boxes else None, "model_reused": True, "device": "cpu",
        "checkpoint": bench.CHECKPOINT, "failure_category": "success" if iou >= 0.5 else "low_overlap",
    }


def test_pixel_box_conversion_to_normalized_xyxy() -> None:
    assert bench.normalized_xyxy([10, 20, 50, 80], 100, 100) == pytest.approx([0.1, 0.2, 0.5, 0.8])
    assert bench.normalized_xyxy([-5, 10, 110, 90], 100, 100) == pytest.approx([0.0, 0.1, 1.0, 0.9])


def test_iou_known_examples() -> None:
    assert bench.box_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert bench.box_iou([0, 0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]) == pytest.approx(1 / 7)
    assert bench.box_iou([0, 0, 0.1, 0.1], [0.9, 0.9, 1, 1]) == 0.0


def test_best_iou_selection_is_not_highest_confidence_selection() -> None:
    predictions = [
        {"bbox_normalized": [0, 0, 0.2, 0.2], "score": 0.99, "label": "bridge"},
        {"bbox_normalized": [0.1, 0.1, 0.5, 0.5], "score": 0.55, "label": "bridge"},
    ]
    best = bench.best_iou_prediction(predictions, [0.1, 0.1, 0.5, 0.5])
    assert best is not None and best["index"] == 1 and best["iou"] == 1.0


def test_null_predictions_count_as_zero_iou() -> None:
    metrics = bench.aggregate([row(1, boxes=0), row(2, iou=1.0)], 2.0)
    assert metrics["null_predictions"] == 1
    assert metrics["null_prediction_rate"] == 0.5
    assert metrics["mean_iou_all"] == 0.5
    assert metrics["mean_iou_non_null"] == 1.0


def test_threshold_metrics() -> None:
    rows = [row(1, iou=0.1), row(2, iou=0.3), row(3, iou=0.6), row(4, iou=0.8)]
    metrics = bench.aggregate(rows, 1.0)
    assert metrics["accuracy_at_0.25"] == 0.75
    assert metrics["accuracy_at_0.50"] == 0.5
    assert metrics["accuracy_at_0.75"] == 0.25


def test_clipped_and_unclipped_grouping() -> None:
    groups = bench.group_metrics([row(1, clipped=True, iou=0.6), row(2, clipped=False, iou=0.2)], lambda item: "clipped" if item["bbox_was_clipped"] else "unclipped")
    mapped = {item["group"]: item for item in groups}
    assert mapped["clipped"]["accuracy_at_0.50"] == 1.0
    assert mapped["unclipped"]["accuracy_at_0.50"] == 0.0


def test_resume_round_trip_skips_completed_smoke_ids(tmp_path: Path) -> None:
    path = tmp_path / "predictions.partial.csv"
    rows = [row(1), row(2)]
    bench.write_predictions(path, rows)
    restored = bench.load_predictions(path)
    records = [{"smoke_id": "1"}, {"smoke_id": "2"}, {"smoke_id": "3"}]
    assert set(restored) == {"1", "2"}
    assert [item["smoke_id"] for item in bench.pending_records(records, restored)] == ["3"]
    assert restored["1"]["predicted_boxes"][0]["bbox_normalized"] == [0.1, 0.1, 0.5, 0.5]


def test_failure_taxonomy_uses_observable_geometry() -> None:
    record = {"ground_truth_bbox": [0.1, 0.1, 0.2, 0.2], "object_class": "bridge", "bbox_was_clipped": False, "is_unique": True, "relative_position": "", "referring_sentence": "The bridge."}
    assert bench.failure_category(record, None) == "null_prediction"
    assert bench.failure_category(record, {"iou": 0.1, "bbox_normalized": [0.0, 0.0, 0.8, 0.8], "label": "bridge"}) == "oversized_box"
    assert bench.failure_category(record, {"iou": 0.1, "bbox_normalized": [0.1, 0.1, 0.12, 0.12], "label": "bridge"}) == "undersized_box"
