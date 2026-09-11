"""Focused tests for the step-100 Grounding Specialist Smoke-100 evaluator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_vrsbench_grounding_specialist_smoke as specialist


def save_checkpoint(path: Path, state: dict | None = None) -> None:
    head = specialist.SatQueryGroundingHead()
    torch.save({
        "model_name": "test", "step": 100, "base_model": specialist.BASE_CHECKPOINT,
        "specialist_state_dict": state if state is not None else head.state_dict(),
    }, path)


def prediction_row(index: int, *, clipped: bool = False, iou: float = 0.6, query_index: int = 4) -> dict:
    return {
        "manifest_index": index, "smoke_id": str(index), "image": f"{index}.png", "image_path": f"{index}.png",
        "object_class": "bridge", "referring_sentence": "The bridge.", "ground_truth_bbox": [0.1, 0.1, 0.5, 0.5],
        "bbox_was_clipped": clipped, "completed": True, "error": None, "selected_query_index": query_index,
        "query_logit": 1.0, "confidence": 0.731, "base_box_cxcywh": [0.3, 0.3, 0.4, 0.4],
        "refined_box_cxcywh": [0.3, 0.3, 0.4, 0.4], "predicted_bbox": [0.1, 0.1, 0.5, 0.5],
        "iou": iou, "predicted_box_area": 0.16, "ground_truth_box_area": 0.16, "area_ratio": 1.0,
        "oversized": False, "undersized": False, "latency_ms": 10.0, "device": "cpu",
        "base_checkpoint": specialist.BASE_CHECKPOINT, "specialist_checkpoint_sha256": "abc",
    }


def test_checkpoint_loading_constructs_exact_head(tmp_path: Path) -> None:
    path = tmp_path / "valid.pt"
    save_checkpoint(path)
    head, payload = specialist.load_specialist_checkpoint(path, torch.device("cpu"))
    assert isinstance(head, specialist.SatQueryGroundingHead)
    assert payload["step"] == 100
    assert len(head.state_dict()) == 12
    assert head.max_box_delta == 0.15


def test_strict_specialist_state_dict_rejects_missing_tensor(tmp_path: Path) -> None:
    state = specialist.SatQueryGroundingHead().state_dict()
    state.pop("query_scorer.0.weight")
    path = tmp_path / "invalid.pt"
    save_checkpoint(path, state)
    with pytest.raises(RuntimeError, match="Missing key"):
        specialist.load_specialist_checkpoint(path, torch.device("cpu"))


def test_cxcywh_to_xyxy_and_clipping() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.2], [0.0, 1.0, 0.4, 0.4]])
    converted = specialist.cxcywh_to_xyxy(boxes)
    assert converted[0].tolist() == pytest.approx([0.3, 0.4, 0.7, 0.6])
    assert converted[1].tolist() == pytest.approx([0.0, 0.8, 0.2, 1.0])


def test_box_sanitation_clamps_centers_and_sizes() -> None:
    sanitized = specialist.sanitize_cxcywh(torch.tensor([[-1.0, 2.0, -0.5, 3.0]]))
    assert sanitized.tolist()[0] == pytest.approx([0.0, 1.0, 0.0001, 1.0])


def test_paired_iou_known_values() -> None:
    first = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]])
    second = torch.tensor([[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 1.0, 1.0]])
    assert specialist.paired_iou(first, second).tolist() == pytest.approx([1 / 7, 1.0])


def test_highest_logit_selects_deployed_query_not_iou() -> None:
    logits = torch.tensor([[0.1, 3.0, 2.0]])
    boxes = torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]])
    indices, scores, selected = specialist.select_highest_logit(logits, boxes)
    assert indices.tolist() == [1]
    assert scores.tolist() == [3.0]
    assert selected.tolist()[0] == pytest.approx([0.4, 0.4, 0.2, 0.2])


def test_query_index_collapse_detection() -> None:
    collapsed = specialist.query_index_analysis([7] * 30 + list(range(70)), [0.1] * 100)
    assert collapsed["collapse_detected"] is True
    assert collapsed["top_1_coverage"] == 0.31  # index 7 also appears in range(70)
    assert "one_query_index_above_25_percent" in collapsed["collapse_reasons"]
    diverse = specialist.query_index_analysis(list(range(100)), [index / 100 for index in range(100)])
    assert diverse["top_5_coverage"] == 0.05


def test_clipped_unclipped_aggregation() -> None:
    groups = specialist.group_metrics(
        [prediction_row(1, clipped=True, iou=0.8), prediction_row(2, clipped=False, iou=0.2)],
        lambda row: "clipped" if row["bbox_was_clipped"] else "unclipped",
    )
    mapped = {item["group"]: item for item in groups}
    assert mapped["clipped"]["accuracy_at_0.50"] == 1.0
    assert mapped["unclipped"]["accuracy_at_0.50"] == 0.0


def test_direct_baseline_comparison() -> None:
    current = {
        "mean_iou": 0.3, "accuracy_at_0.25": 0.5, "accuracy_at_0.50": 0.4,
        "accuracy_at_0.75": 0.2, "null_or_failure_rate": 0.0,
        "average_latency_ms": 100.0, "p95_latency_ms": 120.0,
    }
    baseline = {
        "mean_iou_all": 0.1, "accuracy_at_0.25": 0.2, "accuracy_at_0.50": 0.1,
        "accuracy_at_0.75": 0.05, "null_prediction_rate": 0.6,
        "average_latency_ms": 1000.0, "p95_latency_ms": 1200.0,
    }
    compared = {row["metric"]: row for row in specialist.baseline_comparison(current, baseline)}
    assert compared["mean_iou"]["absolute_change"] == pytest.approx(0.2)
    assert compared["null_rate"]["absolute_change"] == pytest.approx(-0.6)
    assert compared["average_latency_ms"]["absolute_change"] == pytest.approx(-900.0)


def test_resume_round_trip_and_skip_logic(tmp_path: Path) -> None:
    path = tmp_path / "partial.csv"
    rows = [prediction_row(1), prediction_row(2)]
    specialist.write_predictions(path, rows)
    restored = specialist.load_predictions(path)
    records = [{"smoke_id": "1"}, {"smoke_id": "2"}, {"smoke_id": "3"}]
    assert set(restored) == {"1", "2"}
    assert [row["smoke_id"] for row in specialist.pending_records(records, restored)] == ["3"]
    assert restored["1"]["selected_query_index"] == 4


def test_prediction_row_uses_selected_box_once() -> None:
    record = {
        "smoke_id": 1, "image": "one.png", "resolved_image_path": "one.png", "object_class": "bridge",
        "referring_sentence": "The bridge.", "ground_truth_bbox": [0.1, 0.1, 0.5, 0.5],
        "image_width": 100, "image_height": 100, "bbox_was_clipped": False,
    }
    selected = {
        "selected_query_index": 4, "query_logit": 1.0, "confidence": 0.731,
        "base_box_cxcywh": [0.3, 0.3, 0.4, 0.4], "refined_box_cxcywh": [0.3, 0.3, 0.4, 0.4],
        "predicted_bbox": [0.1, 0.1, 0.5, 0.5],
    }
    runner = SimpleNamespace(device=torch.device("cpu"), checkpoint_hash="abc")
    shaped = specialist.shape_row(record, 1, selected, None, 10.0, runner)
    assert shaped["predicted_bbox"] == [0.1, 0.1, 0.5, 0.5]
    assert shaped["iou"] == pytest.approx(1.0)
