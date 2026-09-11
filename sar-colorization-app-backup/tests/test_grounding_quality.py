"""Focused reliability-gate coverage for Grounding DINO localization candidates."""

from __future__ import annotations

import io
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend import app
from satquery_agent.image_ingestion import preview_file_path, remove_preview_url
from satquery_agent.models import ImageFormat, ImageMetadata, Modality
from satquery_agent.reporting import MISSION_STORE
from satquery_agent.specialists.grounder import RemoteSensingGrounder, evaluate_detection_quality


def metadata(width: int = 100, height: int = 100) -> ImageMetadata:
    return ImageMetadata(
        file_id="grounding-quality-test",
        original_name="scene.png",
        safe_name="scene.png",
        format=ImageFormat.PNG,
        mime_type="image/png",
        size_bytes=100,
        width=width,
        height=height,
        band_count=3,
        dtype="uint8",
        is_georeferenced=False,
        color_interpretation=["r", "g", "b"],
    )


class QualityModel:
    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **_kwargs):
        return object()


class QualityProcessor:
    def __init__(self, boxes=None, scores=None, labels=None):
        self.boxes = boxes if boxes is not None else [[20.0, 20.0, 60.0, 60.0]]
        self.scores = scores if scores is not None else [0.75]
        self.labels = labels if labels is not None else ["water body"]

    def __call__(self, *, images, text, return_tensors):
        return {
            "pixel_values": torch.zeros((1, 3, 64, 64), dtype=torch.float32),
            "input_ids": torch.ones((1, 4), dtype=torch.long),
        }

    def post_process_grounded_object_detection(self, *_args, **_kwargs):
        return [{
            "boxes": torch.tensor(self.boxes, dtype=torch.float32),
            "scores": torch.tensor(self.scores, dtype=torch.float32),
            "labels": self.labels,
        }]


def run_grounder(processor: QualityProcessor, query: str = "Highlight the water body."):
    grounder = RemoteSensingGrounder()
    image = Image.new("RGB", (100, 100), (110, 140, 170))
    with patch.object(grounder, "_load_components", return_value=(processor, QualityModel())):
        result = grounder.ground(image, query, metadata(), Modality.OPTICAL, ["r", "g", "b"], "RGB")
    image.close()
    return result


def png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


class GroundingQualityUnitTests(unittest.TestCase):
    def evaluate(self, box, score, target="water body"):
        return evaluate_detection_quality([box], [score], [target], metadata(), target)

    def test_score_below_operational_minimum_is_rejected(self):
        accepted, rejected = self.evaluate([20, 20, 60, 60], 0.449)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0].rejection_reasons, ["alignment_score_below_minimum"])

    def test_localized_area_ratio_above_maximum_is_rejected(self):
        accepted, rejected = self.evaluate([0, 0, 95, 95], 0.8)
        self.assertEqual(accepted, [])
        self.assertIn("localized_box_area_ratio_above_maximum", rejected[0].rejection_reasons)
        self.assertAlmostEqual(rejected[0].box_area_ratio, 0.9025)

    def test_original_full_image_water_candidate_is_rejected_and_auditable(self):
        box = [0.021347, 0.053055, 256.021362, 256.052917]
        accepted, rejected = evaluate_detection_quality(
            [box], [0.37422514], ["water body"], metadata(256, 256), "water body"
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0].bbox_pixels, [0, 0, 256, 256])
        self.assertGreater(rejected[0].box_area_ratio, 0.99)
        self.assertEqual(
            rejected[0].rejection_reasons,
            ["alignment_score_below_minimum", "localized_box_area_ratio_above_maximum"],
        )

    def test_valid_localized_box_is_accepted_with_quality_metrics(self):
        accepted, rejected = self.evaluate([20, 10, 60, 50], 0.75)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0].bbox_pixels, [20, 10, 60, 50])
        self.assertAlmostEqual(accepted[0].quality.box_area_ratio, 0.16)
        self.assertTrue(accepted[0].quality.in_bounds)

    def test_multiple_rejection_reasons_are_retained(self):
        _, rejected = self.evaluate([0, 0, 100, 100], 0.4)
        self.assertEqual(len(rejected[0].rejection_reasons), 2)

    def test_non_finite_score_is_rejected_without_non_json_number(self):
        _, rejected = self.evaluate([20, 20, 60, 60], math.nan)
        self.assertIsNone(rejected[0].score)
        self.assertFalse(rejected[0].quality.finite_score)
        self.assertIn("non_finite_alignment_score", rejected[0].rejection_reasons)

    def test_non_finite_coordinates_are_rejected(self):
        _, rejected = self.evaluate([20, math.inf, 60, 70], 0.8)
        self.assertFalse(rejected[0].quality.finite_coordinates)
        self.assertIsNone(rejected[0].bbox_source_xyxy[1])
        self.assertIn("non_finite_coordinates", rejected[0].rejection_reasons)

    def test_zero_area_box_is_rejected(self):
        _, rejected = self.evaluate([20, 20, 20, 60], 0.8)
        self.assertFalse(rejected[0].quality.positive_area)
        self.assertIn("non_positive_box_area", rejected[0].rejection_reasons)

    def test_accepted_count_excludes_rejected_candidates(self):
        result = run_grounder(QualityProcessor(
            boxes=[[20, 20, 60, 60], [0, 0, 100, 100]],
            scores=[0.75, 0.37422514],
            labels=["water body", "water body"],
        ))
        self.assertEqual(result.accepted_detection_count, 1)
        self.assertEqual(result.rejected_candidate_count, 1)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.accepted_detections, result.detections)
        remove_preview_url(result.annotated_preview_url)

    def test_rejected_candidate_is_not_drawn_in_accepted_preview(self):
        result = run_grounder(QualityProcessor(
            boxes=[[20, 20, 60, 60], [0, 0, 100, 100]],
            scores=[0.75, 0.8],
            labels=["water body", "water body"],
        ))
        path = preview_file_path(result.annotated_preview_url.rsplit("/", 1)[-1])
        with Image.open(path).convert("RGB") as preview:
            cyan_on_outer_border = sum(
                pixel[0] < 80 and pixel[1] > 150 and pixel[2] > 180
                for pixel in ([preview.getpixel((x, 99)) for x in range(100)] + [preview.getpixel((99, y)) for y in range(100)])
            )
        self.assertEqual(cyan_on_outer_border, 0)
        remove_preview_url(result.annotated_preview_url)

    def test_all_rejected_is_completed_empty_result_without_annotated_preview(self):
        result = run_grounder(QualityProcessor(boxes=[[0, 0, 100, 100]], scores=[0.37422514]))
        self.assertEqual(result.detections, [])
        self.assertEqual(result.rejected_candidate_count, 1)
        self.assertIsNone(result.annotated_preview_url)
        self.assertEqual(result.confidence.level.value, "unavailable")
        self.assertEqual(
            result.empty_result_explanation,
            "No reliable localized region was found for 'water body'. The model produced a weak scene-level match rather than a precise object region.",
        )

    def test_valid_building_and_stadium_grounding_still_work(self):
        for query, label in (("Locate the buildings.", "building"), ("Find the stadium.", "stadium")):
            result = run_grounder(QualityProcessor(labels=[label]), query)
            self.assertEqual(len(result.detections), 1)
            self.assertEqual(result.detections[0].label, label)
            remove_preview_url(result.annotated_preview_url)


class GroundingQualityApiAndReportTests(unittest.TestCase):
    def setUp(self):
        MISSION_STORE.clear()
        self.client = TestClient(app)

    def query(self, processor: QualityProcessor):
        grounder = RemoteSensingGrounder()
        with patch.object(grounder, "_load_components", return_value=(processor, QualityModel())), patch(
            "satquery_agent.api.get_grounder", return_value=grounder
        ):
            return self.client.post(
                "/api/agent/query",
                data={"query": "Highlight the water body.", "input_mode": "single", "primary_modality": "optical", "use_cache": "false"},
                files={"primary_image": ("scene.png", png_bytes(), "image/png")},
            )

    def test_original_failure_returns_partial_empty_localization_and_filter_trace(self):
        response = self.query(QualityProcessor(boxes=[[0.021347, 0.053055, 100.021362, 100.052917]], scores=[0.37422514]))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "partial")
        self.assertEqual(body["grounding_result"]["accepted_detection_count"], 0)
        self.assertEqual(body["grounding_result"]["rejected_candidate_count"], 1)
        self.assertEqual(body["evidence"], [])
        self.assertEqual(
            body["answer"],
            "No reliable localized region was found for 'water body'. The model produced a weak scene-level match rather than a precise object region.",
        )
        step = next(item for item in body["execution"]["steps"] if item["tool"] == "grounding_quality_filter")
        self.assertEqual(step["parameters"]["candidate_count"], 1)
        self.assertEqual(step["parameters"]["accepted_count"], 0)
        self.assertEqual(step["parameters"]["rejected_count"], 1)
        self.assertEqual(step["parameters"]["minimum_score"], 0.45)
        self.assertEqual(step["parameters"]["maximum_localized_area_ratio"], 0.85)
        self.assertEqual(step["parameters"]["rejection_reason_counts"]["alignment_score_below_minimum"], 1)

    def test_report_separates_accepted_and_rejected_candidates(self):
        body = self.query(QualityProcessor(
            boxes=[[20, 20, 60, 60], [0, 0, 100, 100]],
            scores=[0.75, 0.37422514],
            labels=["water body", "water body"],
        )).json()
        report = self.client.post("/api/agent/report", json={"request_id": body["request_id"], "formats": ["json", "csv"]})
        self.assertEqual(report.status_code, 200, report.text)
        artifacts = {item["format"]: item for item in report.json()["artifacts"]}
        document = self.client.get(artifacts["json"]["url"]).json()
        grounding = document["authoritative_response"]["grounding_result"]
        self.assertEqual(len(grounding["detections"]), 1)
        self.assertEqual(len(grounding["rejected_candidates"]), 1)
        self.assertEqual(document["statistics"]["grounding_statistics"]["accepted_detection_count"], 1)
        self.assertEqual(document["statistics"]["grounding_statistics"]["rejected_candidate_count"], 1)
        self.assertIn("Operational reliability gates", grounding["operational_threshold_disclaimer"])
        csv_text = self.client.get(artifacts["csv"]["url"]).text
        self.assertIn("grounding_detections", csv_text)
        self.assertIn("grounding_rejected_candidates", csv_text)


if __name__ == "__main__":
    unittest.main()
