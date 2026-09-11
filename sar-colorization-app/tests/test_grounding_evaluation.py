"""Synthetic-fixture tests for the local Grounding reliability evaluator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from satquery_agent.grounding_evaluation import (
    AREA_RATIO_GRID,
    SCORE_GRID,
    CandidateInference,
    GroundingEvaluationError,
    ModelCandidate,
    aggregate_grounding_evaluation,
    box_iou,
    center_point_in_reference,
    evaluate_grounding_dataset,
    evaluate_threshold_grid,
    load_grounding_evaluation_dataset,
    write_grounding_evaluation_outputs,
)
from satquery_agent.specialists.grounder import DEFAULT_RELIABILITY_POLICY


def sample(
    sample_id: str,
    target: str = "stadium",
    boxes=None,
    target_present: bool = True,
    image: str | None = None,
):
    return {
        "id": sample_id,
        "image": image or f"images/{sample_id}.png",
        "target": target,
        "query": f"Locate the {target}.",
        "target_present": target_present,
        "boxes": boxes if boxes is not None else [[20, 20, 60, 60]],
        "source": "manual",
        "notes": None,
    }


class FixtureProvider:
    def __init__(self, predictions):
        self.predictions = predictions

    def __call__(self, item):
        return CandidateInference(candidates=tuple(self.predictions.get(item.sample_id, ())), device="cpu")


class GroundingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "grounding_eval"
        (self.root / "images").mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def write_dataset(self, samples):
        for item in samples:
            image_path = self.root / item["image"]
            if item.get("create_image", True):
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 100), (90, 120, 150)).save(image_path)
            item.pop("create_image", None)
        (self.root / "annotations.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")
        return load_grounding_evaluation_dataset(self.root)

    def evaluate(self, rows, predictions):
        return evaluate_grounding_dataset(self.write_dataset(rows), FixtureProvider(predictions))

    def test_valid_dataset_loading(self):
        dataset = self.write_dataset([sample("stadium_001")])
        self.assertEqual(dataset.samples[0].sample_id, "stadium_001")
        self.assertEqual(dataset.samples[0].annotation_source, "manual")
        self.assertEqual(dataset.samples[0].width, 100)

    def test_invalid_annotation_rejection(self):
        with self.assertRaisesRegex(GroundingEvaluationError, "zero- or negative-area"):
            self.write_dataset([sample("bad_box", boxes=[[20, 20, 20, 60]])])

    def test_missing_image_handling(self):
        with self.assertRaisesRegex(GroundingEvaluationError, "missing image"):
            self.write_dataset([{**sample("missing"), "create_image": False}])

    def test_positive_sample_logic(self):
        report = self.evaluate(
            [sample("positive")],
            {"positive": [ModelCandidate(0.8, (20, 20, 60, 60), "stadium")]},
        )
        self.assertEqual(report["metrics"]["positive_sample_count"], 1)
        self.assertEqual(report["metrics"]["true_positive_count_iou_25"], 1)

    def test_negative_sample_logic(self):
        report = self.evaluate(
            [sample("negative", boxes=[], target_present=False)],
            {"negative": [ModelCandidate(0.4, (10, 10, 50, 50), "stadium")]},
        )
        self.assertEqual(report["metrics"]["negative_sample_count"], 1)
        self.assertEqual(report["metrics"]["correct_rejection_count"], 1)
        self.assertEqual(report["metrics"]["weak_rejected_negative_count"], 1)

    def test_one_reference_box(self):
        report = self.evaluate(
            [sample("one_box", boxes=[[10, 10, 40, 40]])],
            {"one_box": [ModelCandidate(0.8, (10, 10, 40, 40), "stadium")]},
        )
        candidate = report["samples"][0]["candidates"][0]
        self.assertEqual(candidate["reference_ious"], [1.0])
        self.assertEqual(candidate["matched_reference_index"], 0)

    def test_multiple_reference_boxes(self):
        report = self.evaluate(
            [sample("multiple", target="building", boxes=[[10, 10, 30, 30], [60, 60, 90, 90]])],
            {"multiple": [ModelCandidate(0.8, (60, 60, 90, 90), "building")]},
        )
        candidate = report["samples"][0]["candidates"][0]
        self.assertEqual(len(candidate["reference_ious"]), 2)
        self.assertEqual(candidate["matched_reference_index"], 1)

    def test_iou_calculation(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(box_iou((0, 0, 2, 2), (3, 3, 4, 4)), 0.0)

    def test_iou_threshold_evaluation(self):
        report = self.evaluate(
            [sample("threshold", boxes=[[0, 0, 40, 40]])],
            {"threshold": [ModelCandidate(0.8, (0, 0, 40, 20), "stadium")]},
        )
        candidate = report["samples"][0]["candidates"][0]
        self.assertTrue(candidate["meets_iou_25"])
        self.assertTrue(candidate["meets_iou_50"])
        self.assertEqual(report["metrics"]["true_positive_count_iou_50"], 1)

    def test_center_inclusion(self):
        self.assertTrue(center_point_in_reference((20, 20, 40, 40), (10, 10, 50, 50)))
        self.assertFalse(center_point_in_reference((70, 70, 90, 90), (10, 10, 50, 50)))

    def test_accepted_true_positive(self):
        report = self.evaluate(
            [sample("true_positive")],
            {"true_positive": [ModelCandidate(0.75, (22, 22, 58, 58), "stadium")]},
        )
        self.assertEqual(report["samples"][0]["candidates"][0]["outcome"], "accepted_true_positive")
        self.assertEqual(report["metrics"]["accepted_precision_iou_25"], 1.0)

    def test_accepted_false_localization(self):
        report = self.evaluate(
            [sample("false_localization")],
            {"false_localization": [ModelCandidate(0.8, (70, 70, 90, 90), "stadium")]},
        )
        self.assertEqual(report["samples"][0]["candidates"][0]["outcome"], "accepted_false_localization")
        self.assertEqual(report["metrics"]["false_localization_count"], 1)

    def test_correctly_rejected_weak_candidate(self):
        report = self.evaluate(
            [sample("weak")],
            {"weak": [ModelCandidate(0.4, (20, 20, 60, 60), "stadium")]},
        )
        candidate = report["samples"][0]["candidates"][0]
        self.assertFalse(candidate["accepted"])
        self.assertIn("alignment_score_below_minimum", candidate["rejection_reasons"])

    def test_missed_target(self):
        report = self.evaluate([sample("missed")], {"missed": []})
        self.assertTrue(report["samples"][0]["missed_target"])
        self.assertEqual(report["metrics"]["missed_target_count"], 1)
        self.assertEqual(report["metrics"]["no_prediction_count"], 1)

    def test_full_frame_rejection(self):
        report = self.evaluate(
            [sample("full_frame", target="water body")],
            {"full_frame": [ModelCandidate(0.8, (0, 0, 100, 100), "water body")]},
        )
        candidate = report["samples"][0]["candidates"][0]
        self.assertTrue(candidate["full_frame_prediction"])
        self.assertFalse(candidate["accepted"])
        self.assertEqual(report["metrics"]["full_frame_acceptance_count"], 0)

    def test_per_target_aggregation_and_unknown_category(self):
        rows = [sample("stadium_target"), sample("unknown_target", target="solar array")]
        predictions = {
            "stadium_target": [ModelCandidate(0.8, (20, 20, 60, 60), "stadium")],
            "unknown_target": [ModelCandidate(0.8, (20, 20, 60, 60), "solar array")],
        }
        report = self.evaluate(rows, predictions)
        self.assertEqual([item["target"] for item in report["per_target"]], ["solar array", "stadium"])

    def test_threshold_grid_output(self):
        report = self.evaluate(
            [sample("grid")],
            {"grid": [ModelCandidate(0.42, (20, 20, 60, 60), "stadium")]},
        )
        rows = evaluate_threshold_grid(report)
        self.assertEqual(len(rows), len(SCORE_GRID) * len(AREA_RATIO_GRID))
        low = next(row for row in rows if row["minimum_score"] == 0.40 and row["maximum_area_ratio"] == 0.85)
        high = next(row for row in rows if row["minimum_score"] == 0.45 and row["maximum_area_ratio"] == 0.85)
        self.assertEqual(low["accepted_count"], 1)
        self.assertEqual(high["accepted_count"], 0)

    def test_null_unavailable_metric_handling(self):
        metrics = aggregate_grounding_evaluation([])
        self.assertIsNone(metrics["accepted_precision_iou_25"])
        self.assertIsNone(metrics["accepted_recall_iou_25"])
        self.assertIsNone(metrics["score_distribution"]["mean"])

    def test_analysis_override_does_not_mutate_production_thresholds(self):
        before = (DEFAULT_RELIABILITY_POLICY.minimum_alignment_score, DEFAULT_RELIABILITY_POLICY.maximum_localized_area_ratio)
        report = self.evaluate(
            [sample("override")],
            {"override": [ModelCandidate(0.32, (20, 20, 60, 60), "stadium")]},
        )
        dataset = self.write_dataset([sample("override_two")])
        evaluate_grounding_dataset(dataset, FixtureProvider({}), minimum_score=0.30, maximum_area_ratio=0.60)
        after = (DEFAULT_RELIABILITY_POLICY.minimum_alignment_score, DEFAULT_RELIABILITY_POLICY.maximum_localized_area_ratio)
        self.assertEqual(before, after)
        self.assertEqual(report["production_gate_reference"]["minimum_alignment_score"], 0.45)

    def test_safe_exported_paths_and_complete_artifacts(self):
        report = self.evaluate(
            [sample("safe_export")],
            {"safe_export": [ModelCandidate(0.8, (20, 20, 60, 60), "stadium")]},
        )
        output = Path(self.temporary.name) / "artifacts"
        names = write_grounding_evaluation_outputs(report, output)
        self.assertEqual(len(names), 7)
        self.assertTrue(all((output / name).is_file() for name in names))
        exported = "\n".join((output / name).read_text(encoding="utf-8") for name in names)
        self.assertNotIn(str(self.root), exported)
        self.assertNotIn("safe_export.png", exported)
        self.assertIn("safe_export", exported)


if __name__ == "__main__":
    unittest.main()
