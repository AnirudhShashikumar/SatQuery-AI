"""Focused tests for the durable official RSVQA-LR evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from scripts import run_rsvqa_full as full


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.posts: list[dict] = []

    def post(self, endpoint, **kwargs):
        self.posts.append({"endpoint": endpoint, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize(
    ("value", "task", "exact", "overflow"),
    [
        (" YES! ", "presence", "yes", "yes"),
        ("urban.", "rural_urban", "urban", "urban"),
        ("200", "count", "200", "200"),
        ("201", "count", "201", "201+"),
        ("403", "count", "403", "201+"),
        ("201+", "count", "201+", "201+"),
        (None, "count", None, None),
    ],
)
def test_exact_and_overflow_normalization(value, task, exact, overflow) -> None:
    assert full.normalize_answer(value, task) == exact
    assert full.normalize_answer(value, task, overflow_aware=True) == overflow


def test_official_exports_are_joined_without_changing_labels(tmp_path: Path) -> None:
    questions = {"questions": [
        {"id": 10, "img_id": 7, "type": "count", "question": "How many roads?", "answers_ids": [10], "active": True},
        {"id": 11, "img_id": 7, "type": "presence", "question": "Is there water?", "answers_ids": [11], "active": True},
    ]}
    answers = {"answers": [
        {"id": 10, "question_id": 10, "answer": "403", "active": True},
        {"id": 11, "question_id": 11, "answer": "yes", "active": True},
    ]}
    images = {"images": [{"id": 7, "original_name": "official.tif", "sensor": "S2", "type": "RGB", "res_x": "10m", "res_y": "10m", "active": True}]}
    paths = []
    for name, value in (("questions.json", questions), ("answers.json", answers), ("images.json", images)):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "manifest.jsonl"
    with patch.object(full, "EXPECTED_QUESTIONS", 2), patch.object(full, "EXPECTED_IMAGES", 1):
        provenance = full.build_manifest_from_official_exports(*paths, output)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert provenance["records"] == 2
    assert records[0]["ground_truth"] == "403"
    assert records[0]["question"] == "How many roads?"


def _config(tmp_path: Path, **updates) -> full.FullBenchmarkConfig:
    values = dict(
        manifest=tmp_path / "manifest.jsonl",
        image_root=tmp_path,
        endpoint="http://example.test/api/analysis/image",
        output_dir=tmp_path / "output",
        timeout=1.0,
        checkpoint_every=2,
        max_retries=3,
    )
    values.update(updates)
    return full.FullBenchmarkConfig(**values)


def test_endpoint_request_never_sends_ground_truth_or_question_type(tmp_path: Path) -> None:
    png = tmp_path / "7.png"
    png.write_bytes(b"png")
    response = FakeResponse(200, {"answer": "yes", "model_used": full.MODEL_USED})
    session = FakeSession([response])
    payload, error, _, _, attempts, _ = full.post_question(
        {"question": "Is there water?", "ground_truth": "yes", "question_type": "presence"},
        png,
        _config(tmp_path),
        session,
    )
    assert error is None and payload["answer"] == "yes" and attempts == 1
    assert session.posts[0]["data"] == {
        "analysis_type": "ground_truth",
        "model_name": "satquery-agent",
        "question": "Is there water?",
    }


def test_transient_failures_retry_but_validation_error_does_not(tmp_path: Path) -> None:
    png = tmp_path / "7.png"
    png.write_bytes(b"png")
    transient = FakeSession([FakeResponse(503, {"detail": "busy"}), FakeResponse(200, {"answer": "yes"})])
    with patch.object(full.time, "sleep"):
        payload, error, _, _, attempts, _ = full.post_question({"question": "Q"}, png, _config(tmp_path), transient)
    assert payload == {"answer": "yes"} and error is None and attempts == 2
    deterministic = FakeSession([FakeResponse(400, {"detail": "invalid"}), FakeResponse(200, {})])
    payload, error, _, status, attempts, _ = full.post_question({"question": "Q"}, png, _config(tmp_path), deterministic)
    assert payload is None and error == "invalid" and status == 400 and attempts == 1


def _row(index: int, task: str, truth: str, prediction: str | None, confidence: float = 0.8) -> dict:
    exact = prediction == truth
    overflow_truth = "201+" if task == "count" and truth.isdigit() and int(truth) > 200 else truth
    return {
        "manifest_index": index, "question_id": str(index), "image_id": str(index),
        "source_image_path": f"{index}.tif", "converted_image_path": f"{index}.png",
        "question": "How many roads?" if task == "count" else "Is there water?",
        "question_type": task, "ground_truth": truth,
        "normalized_ground_truth_exact": truth,
        "normalized_ground_truth_overflow": overflow_truth,
        "answer": prediction, "normalized_prediction": prediction,
        "exact_correct": exact, "overflow_correct": prediction == overflow_truth,
        "outcome": "CORRECT" if exact else "WRONG", "completed": True,
        "endpoint_error": None, "error_stage": None, "http_status": 200, "attempts": 1,
        "confidence": confidence, "task": task, "status": "success", "result_status": "COMPLETED",
        "processing_time_ms": 10 + index, "client_elapsed_ms": 12 + index,
        "effective_latency_ms": 10 + index, "model_used": full.MODEL_USED,
        "specialist_used": True, "fallback_used": False, "cached": False,
        "warnings": [], "conversion_cache_hit": True,
    }


def test_checkpoint_round_trip_and_count_metrics(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.manifest.write_text("{}\n", encoding="utf-8")
    rows = [_row(1, "count", "0", "0"), _row(2, "count", "403", "201+"), _row(3, "count", "5", "7")]
    state = {"status": "running"}
    full.checkpoint(config, {row["question_id"]: row for row in rows}, state)
    restored = full.load_predictions(config.partial_predictions)
    assert len(restored) == 3 and restored["1"]["exact_correct"] is True
    metrics = full.count_metrics(rows)
    assert metrics["exact_count_accuracy"] == pytest.approx(1 / 3)
    assert metrics["overflow_aware_accuracy"] == pytest.approx(2 / 3)
    assert metrics["mean_absolute_error"] == pytest.approx(1.0)
    assert metrics["within_2"] == 1.0
    assert config.partial_errors.is_file() and config.run_state.is_file()


def test_charts_write_png_and_svg(tmp_path: Path) -> None:
    full.write_bar_chart(tmp_path, "accuracy", "Accuracy", ["presence", "count"], [0.9, 0.5])
    assert (tmp_path / "accuracy.png").stat().st_size > 0
    assert "<svg" in (tmp_path / "accuracy.svg").read_text(encoding="utf-8")


def test_error_analysis_handles_normal_wrong_answers_without_endpoint_errors(tmp_path: Path) -> None:
    wrong = _row(1, "count", "5", "7")
    wrong["endpoint_error"] = None
    analysis = full.write_error_analysis([wrong], tmp_path)
    assert analysis["incorrect_or_error_records"] == 1
    assert analysis["preprocessing_issues"] == 0
    assert analysis["dominant_count_failure_prediction"] == "7"


def test_top_class_confusion_matrix_keeps_unlisted_ground_truth_rows(tmp_path: Path) -> None:
    rows = [_row(1, "count", "0", "0"), _row(2, "count", "403", "201+")]
    path = tmp_path / "matrix.csv"
    full.write_matrix(path, rows, ["0", "<null>", "<error>"])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("<other>,1,") for line in lines)


def test_reproducibility_command_quotes_paths_with_spaces(tmp_path: Path) -> None:
    config = _config(tmp_path / "directory with spaces")
    command = full.reproducibility_command(config)
    assert "'" in command
    assert "--checkpoint-every 2" in command
