"""Focused metric, adapter, and evidence-gating tests for evaluation runners."""

import json

import numpy as np
import pytest
from PIL import Image

from scripts.calibrate_confidence import calibration_metrics
from scripts.evaluate_captions import bleu_n, rouge_l
from scripts.evaluate_cdvqa import discover_cdvqa, normalize_answer
from scripts.evaluate_changerex_levircd import discover_split
from scripts.evaluate_optical_sar import fusion_gain
from scripts.evaluate_sar_translation import lab_prediction_to_rgb, rgb_metrics
from scripts.evaluation_common import binary_confusion, binary_metrics
from scripts.evaluation_common import benchmark_record
from scripts.benchmark_data import validate_result


def test_binary_change_metrics_use_global_confusion() -> None:
    truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    counts = binary_confusion(prediction, truth)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert binary_metrics(counts)["iou"] == pytest.approx(1 / 3)


def test_levir_discovery_rejects_incomplete_split(tmp_path) -> None:
    for folder in ("A", "B", "label"):
        (tmp_path / "test" / folder).mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(tmp_path / "test" / "A" / "one.png")
    with pytest.raises(RuntimeError, match="Incomplete"):
        discover_split(tmp_path, "test")


def test_cdvqa_adapter_preserves_question_and_official_split(tmp_path) -> None:
    (tmp_path / "images").mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (4, 4)).save(tmp_path / "images" / name)
    question = "Has the built-up area increased?"
    (tmp_path / "test.json").write_text(json.dumps([{"id": 7, "question": question, "answer": "Yes.", "image1": "a.png", "image2": "b.png", "category": "built_up"}]), encoding="utf-8")
    rows = discover_cdvqa(tmp_path, "test")
    assert rows[0]["question"] == question
    assert normalize_answer(rows[0]["answer"]) == "yes"


def test_fusion_gain_requires_labels_on_same_samples() -> None:
    assert fusion_gain([{"reference_answer": None}]) is None
    rows = [
        {"reference_answer": "yes", "optical_correct": True, "sar_correct": False, "fused_correct": True},
        {"reference_answer": "no", "optical_correct": False, "sar_correct": False, "fused_correct": True},
    ]
    assert fusion_gain(rows) == pytest.approx(0.5)


def test_sarfusionformer_lab_is_converted_before_rgb_metrics() -> None:
    lab = np.zeros((3, 8, 8), dtype=np.float32)
    lab[0] = 1.0
    lab[1:] = 128 / 255
    rgb = lab_prediction_to_rgb(lab)
    assert rgb.shape == (8, 8, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1
    assert rgb_metrics(rgb, rgb)["mae"] == 0.0


def test_caption_metrics_are_reference_based() -> None:
    assert bleu_n("water body visible", ["a water body is visible"], 1) > 0
    assert rouge_l("water body visible", ["a water body is visible"]) > 0


def test_confidence_calibration_and_insufficient_evidence() -> None:
    assert calibration_metrics([{"confidence": 0.9, "correct": True}], minimum_samples=10)["ece"] is None
    rows = [{"confidence": value / 10, "correct": value >= 5} for value in range(10)]
    result = calibration_metrics(rows, bins=5, minimum_samples=10)
    assert result["ece"] is not None
    assert result["brier_score"] is not None
    assert len(result["bins"]) == 5


def test_generated_benchmark_record_conforms_to_canonical_contract() -> None:
    record = benchmark_record(
        benchmark_id="fixture.valid", specialist_id="fixture", display_name="Fixture",
        task="test", model_name="Fixture model", model_version="1", checkpoint=None,
        checkpoint_sha256=None, dataset="fixture", split="test", sample_count=None,
        status="unavailable", metrics=[{"name": "accuracy", "value": None, "unit": "ratio", "primary": True}],
        performance={}, artifacts=["report.md"], limitations=["Fixture only."],
    )
    validate_result(record)
    assert record["artifacts"]["report"] == "report.md"
    assert record["metrics"][0]["id"] == "accuracy"
