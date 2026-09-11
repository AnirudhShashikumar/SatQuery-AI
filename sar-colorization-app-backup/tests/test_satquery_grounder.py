"""Unit and opt-in integration coverage for local Grounding DINO grounding."""

from __future__ import annotations

import os
import io
import json
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend import app
from satquery_agent.image_ingestion import preview_file_path, remove_preview_url
from satquery_agent.models import AgentQueryRequest, ImageFormat, ImageMetadata, InputMode, Modality, TaskType
from satquery_agent.reporting import MISSION_STORE
from satquery_agent.router import route_query
from satquery_agent.specialists.grounder import (
    CHECKPOINT,
    MODEL_LICENSE,
    MODEL_SOURCE,
    DEFAULT_CONFIG,
    GrounderError,
    GroundingConfig,
    RemoteSensingGrounder,
    normalize_grounding_target,
    sanitize_detections,
    safe_grounding_parameters,
)


def metadata(*, georeferenced: bool = False, bands: int = 3, width: int = 100, height: int = 100) -> ImageMetadata:
    return ImageMetadata(
        file_id="grounding-test",
        original_name="scene.png",
        safe_name="scene.png",
        format=ImageFormat.PNG,
        mime_type="image/png",
        size_bytes=100,
        width=width,
        height=height,
        band_count=bands,
        dtype="uint8",
        crs="EPSG:4326" if georeferenced else None,
        transform=[1.0, 0.0, 70.0, 0.0, -1.0, 20.0] if georeferenced else None,
        is_georeferenced=georeferenced,
        color_interpretation=["r", "g", "b"][:bands],
    )


class FakeProcessor:
    load_count = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.load_count += 1
        return cls()

    def __call__(self, *, images, text, return_tensors):
        self.prompt = text
        return {
            "pixel_values": torch.zeros((1, 3, 32, 48), dtype=torch.float32),
            "input_ids": torch.ones((1, 8), dtype=torch.long),
        }

    def post_process_grounded_object_detection(self, *_args, **_kwargs):
        return [{
            "boxes": torch.tensor([[10.0, 12.0, 70.0, 76.0], [82.0, 82.0, 120.0, 115.0]]),
            "scores": torch.tensor([0.74, 0.56]),
            "labels": ["building", "building"],
        }]


class EmptyProcessor(FakeProcessor):
    def post_process_grounded_object_detection(self, *_args, **_kwargs):
        return [{"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": []}]


class FakeModel:
    load_count = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.load_count += 1
        return cls()

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **_kwargs):
        return object()


class GrounderUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeProcessor.load_count = 0
        FakeModel.load_count = 0

    def test_target_phrase_extraction_and_rejection(self):
        cases = {
            "Highlight the water body.": "water body",
            "Show me the buildings.": "building",
            "Locate the roads.": "road",
            "Mark all aircraft.": "aircraft",
            "Where is the stadium?": "stadium",
            "Find the bridge.": "bridge",
            "Highlight vegetation.": "vegetation",
        }
        for query, expected in cases.items():
            self.assertEqual(normalize_grounding_target(query), expected)
        for query in ("", "Why is the road here?", "How many buildings are there?", "Highlight ownership.", "Is water present?"):
            with self.assertRaises(GrounderError):
                normalize_grounding_target(query)

    def test_target_phrase_strips_scene_suffix_and_canonicalizes_plural(self):
        self.assertEqual(normalize_grounding_target("Show the bridges in this image."), "bridge")
        self.assertEqual(normalize_grounding_target("Locate all water bodies within the scene."), "water body")

    def test_public_configuration_is_bounded_and_contains_no_paths(self):
        parameters = safe_grounding_parameters()
        self.assertEqual(CHECKPOINT, "IDEA-Research/grounding-dino-tiny")
        self.assertEqual(MODEL_LICENSE, "Apache-2.0")
        self.assertTrue(MODEL_SOURCE.startswith("https://huggingface.co/"))
        self.assertGreater(parameters["box_threshold"], 0)
        self.assertLessEqual(parameters["maximum_detections"], 20)
        self.assertNotIn("cache", json.dumps(parameters).lower())

    def test_router_prioritizes_supported_grounding_but_not_generic_vqa(self):
        for query in (
            "Highlight the water bodies.", "Locate all buildings.", "Show me where the roads are.",
            "Mark the aircraft.", "Where is the stadium?", "Identify the region containing vegetation.",
        ):
            plan = route_query(AgentQueryRequest(query=query, input_mode=InputMode.SINGLE, primary_modality=Modality.OPTICAL, has_primary_image=True, has_secondary_image=False))
            self.assertEqual(plan.detected_task, TaskType.GROUNDING, query)
        plan = route_query(AgentQueryRequest(query="Where is the largest changed region?", input_mode=InputMode.BI_TEMPORAL, primary_modality=Modality.OPTICAL, secondary_modality=Modality.OPTICAL, has_primary_image=True, has_secondary_image=True))
        self.assertEqual(plan.detected_task, TaskType.CHANGE_VQA)
        plan = route_query(AgentQueryRequest(query="Is water present?", input_mode=InputMode.SINGLE, primary_modality=Modality.OPTICAL, has_primary_image=True, has_secondary_image=False))
        self.assertNotEqual(plan.detected_task, TaskType.GROUNDING)

    def test_postprocessing_clamps_sorts_nms_rejects_and_caps(self):
        config = GroundingConfig(box_threshold=.35, text_threshold=.25, nms_iou_threshold=.5, maximum_detections=2)
        result = sanitize_detections(
            boxes=[[90, 90, 130, 120], [10, 10, 10, 30], [-10, -5, 50, 50], [0, 0, 48, 48], [60, 5, 80, 25]],
            scores=[.7, .99, .9, .8, .6],
            labels=["road"] * 5,
            metadata=metadata(),
            target_phrase="road",
            config=config,
        )
        self.assertEqual([item.score for item in result], [.9, .7])
        self.assertEqual(result[0].bbox_pixels, [0, 0, 50, 50])
        self.assertEqual(result[1].bbox_pixels, [90, 90, 100, 100])
        self.assertTrue(all(item.mask_url is None for item in result))

    def test_postprocessing_drops_below_threshold_and_normalizes_coordinates(self):
        result = sanitize_detections([[5, 10, 25, 30], [1, 1, 9, 9]], [.8, DEFAULT_CONFIG.box_threshold - .01], ["road", "road"], metadata(), "road")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox_normalized, [.05, .1, .25, .3])

    def test_world_coordinates_only_when_georeferenced(self):
        plain = sanitize_detections([[10, 20, 30, 40]], [.8], ["bridge"], metadata(), "bridge")
        world = sanitize_detections([[10, 20, 30, 40]], [.8], ["bridge"], metadata(georeferenced=True), "bridge")
        self.assertIsNone(plain[0].bbox_world)
        self.assertEqual(world[0].bbox_world, [80.0, -20.0, 100.0, 0.0])
        self.assertEqual(world[0].crs, "EPSG:4326")

    def test_health_does_not_force_load(self):
        grounder = RemoteSensingGrounder()
        self.assertEqual(grounder.health().status, "unloaded")
        self.assertEqual(FakeModel.load_count, 0)

    def test_load_failure_is_sanitized_and_sticky(self):
        grounder = RemoteSensingGrounder()
        with patch.object(grounder, "_load_components", side_effect=RuntimeError("secret /private/model/path")):
            with self.assertRaisesRegex(GrounderError, "could not be loaded"):
                grounder.load()
        health = grounder.health()
        self.assertEqual(health.status, "failed")
        self.assertEqual(health.error, "Checkpoint loading failed.")
        self.assertNotIn("private", health.error.lower())

    def test_lazy_load_reuse_detection_preview_provenance_and_confidence(self):
        grounder = RemoteSensingGrounder()
        image = Image.new("RGB", (100, 100), (100, 130, 160))
        loader = lambda _local_only: (FakeProcessor.from_pretrained(), FakeModel.from_pretrained())
        with patch.object(grounder, "_load_components", side_effect=loader):
            first = grounder.ground(image, "Locate the buildings.", metadata(), Modality.OPTICAL, ["r", "g", "b"], "RGB visualization")
            second = grounder.ground(image, "Locate the buildings.", metadata(), Modality.OPTICAL, ["r", "g", "b"], "RGB visualization")
        self.assertEqual(FakeProcessor.load_count, 1)
        self.assertEqual(FakeModel.load_count, 1)
        self.assertFalse(first.model_reused)
        self.assertTrue(second.model_reused)
        self.assertEqual([item.score for item in first.detections], [.74, .56])
        self.assertEqual(first.detections[1].bbox_pixels, [82, 82, 100, 100])
        self.assertFalse(first.model.remote_sensing_adapted)
        self.assertIn("not calibrated", first.confidence.reason.lower())
        self.assertIsNotNone(first.annotated_preview_url)
        self.assertIsNotNone(preview_file_path(first.annotated_preview_url.rsplit("/", 1)[-1]))
        remove_preview_url(first.annotated_preview_url)
        remove_preview_url(second.annotated_preview_url)
        image.close()

    def test_processor_receives_canonical_lowercase_period_prompt(self):
        grounder = RemoteSensingGrounder()
        image = Image.new("RGB", (100, 100))
        loader = lambda _local_only: (FakeProcessor(), FakeModel())
        with patch.object(grounder, "_load_components", side_effect=loader):
            result = grounder.ground(image, "Show me the BUILDINGS!", metadata(), Modality.OPTICAL, ["r", "g", "b"], "RGB")
        self.assertEqual(grounder._processor.prompt, "building.")
        self.assertEqual((result.input.model_input_width, result.input.model_input_height), (48, 32))
        remove_preview_url(result.annotated_preview_url)
        image.close()

    def test_annotated_preview_uses_uuid_filename(self):
        grounder = RemoteSensingGrounder()
        image = Image.new("RGB", (100, 100))
        with patch.object(grounder, "_load_components", return_value=(FakeProcessor(), FakeModel())):
            result = grounder.ground(image, "Find the bridge.", metadata(), Modality.OPTICAL, ["r", "g", "b"], "RGB")
        filename = result.annotated_preview_url.rsplit("/", 1)[-1]
        self.assertRegex(filename, r"^[0-9a-f]{32}\.png$")
        remove_preview_url(result.annotated_preview_url)
        image.close()

    def test_no_detection_is_honest_and_has_no_preview(self):
        grounder = RemoteSensingGrounder()
        image = Image.new("RGB", (100, 100))
        loader = lambda _local_only: (EmptyProcessor.from_pretrained(), FakeModel.from_pretrained())
        with patch.object(grounder, "_load_components", side_effect=loader):
            result = grounder.ground(image, "Find the stadium.", metadata(), Modality.OPTICAL, ["r", "g", "b"], "RGB visualization")
        self.assertEqual(result.detections, [])
        self.assertIsNone(result.annotated_preview_url)
        self.assertIsNone(result.confidence.score)
        self.assertIn("no box was invented", result.confidence.reason.lower())
        image.close()

    def test_sar_and_non_rgb_inputs_are_rejected_before_loading(self):
        grounder = RemoteSensingGrounder()
        image = Image.new("RGB", (100, 100))
        with self.assertRaisesRegex(GrounderError, "SAR grounding"):
            grounder.ground(image, "Find the bridge.", metadata(), Modality.SAR, ["band_1"], "SAR")
        with self.assertRaisesRegex(GrounderError, "at least three bands"):
            grounder.ground(image, "Find the bridge.", metadata(bands=1), Modality.OPTICAL, ["band_1"], "grayscale")
        self.assertEqual(grounder.health().status, "unloaded")
        image.close()

    @unittest.skipUnless(os.getenv("SATQUERY_RUN_REAL_GROUNDER") == "1", "real Grounding DINO checkpoint test is opt-in")
    def test_real_cached_checkpoint_contract(self):
        sample = Image.open("satquery_agent/demo_samples/single-optical.png").convert("RGB")
        grounder = RemoteSensingGrounder()
        result = grounder.ground(sample, "Locate the buildings.", metadata(width=sample.width, height=sample.height), Modality.OPTICAL, ["r", "g", "b"], "RGB visualization")
        self.assertEqual(result.model.checkpoint, "IDEA-Research/grounding-dino-tiny")
        self.assertFalse(result.model.remote_sensing_adapted)
        self.assertTrue(result.detections, "The approved local optical sample should produce at least one real candidate building box.")
        self.assertTrue(all(0 <= item.bbox_pixels[0] <= item.bbox_pixels[2] <= sample.width and 0 <= item.bbox_pixels[1] <= item.bbox_pixels[3] <= sample.height for item in result.detections))
        if result.annotated_preview_url:
            remove_preview_url(result.annotated_preview_url)
        sample.close()


def png_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


class GroundingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        MISSION_STORE.clear()
        self.client = TestClient(app)
        self.array = np.full((96, 96, 3), (128, 150, 172), dtype=np.uint8)
        self.array[14:70, 18:74] = (210, 205, 190)

    def query(self, query: str, modality: str = "optical", grounder: RemoteSensingGrounder | None = None):
        grounder = grounder or RemoteSensingGrounder()
        loader = lambda _local_only: (FakeProcessor.from_pretrained(), FakeModel.from_pretrained())
        with patch.object(grounder, "_load_components", side_effect=loader), patch("satquery_agent.api.get_grounder", return_value=grounder):
            return self.client.post(
                "/api/agent/query",
                data={"query": query, "input_mode": "single", "primary_modality": modality, "use_cache": "false"},
                files={"primary_image": ("scene.png", png_bytes(self.array), "image/png")},
            )

    def test_api_returns_typed_detections_preview_trace_and_safe_parameters(self):
        response = self.query("Locate all buildings.")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["task"], "grounding")
        self.assertEqual(result["status"], "success")
        grounding = result["grounding_result"]
        self.assertEqual(grounding["target_phrase"], "building")
        self.assertEqual(len(grounding["detections"]), 2)
        self.assertIsNone(grounding["detections"][0]["mask_url"])
        self.assertTrue(grounding["annotated_preview_url"].startswith("/api/agent/previews/"))
        self.assertEqual(self.client.get(grounding["annotated_preview_url"]).status_code, 200)
        tools = [step["tool"] for step in result["execution"]["steps"]]
        for expected in ("target_phrase_extraction", "grounding_image_preparation", "grounder_model_load", "grounding_inference", "grounding_postprocessing", "grounding_preview_generation"):
            self.assertIn(expected, tools)
        params = result["execution"]["permitted_parameters"]
        self.assertEqual(params["checkpoint"], "IDEA-Research/grounding-dino-tiny")
        self.assertFalse(params["mask_refinement"])
        self.assertNotIn("cache_dir", json.dumps(result))

    def test_api_empty_detection_is_partial_and_does_not_invent_preview(self):
        grounder = RemoteSensingGrounder()
        loader = lambda _local_only: (EmptyProcessor.from_pretrained(), FakeModel.from_pretrained())
        with patch.object(grounder, "_load_components", side_effect=loader), patch("satquery_agent.api.get_grounder", return_value=grounder):
            response = self.client.post(
                "/api/agent/query",
                data={"query": "Find the stadium.", "input_mode": "single", "primary_modality": "optical", "use_cache": "false"},
                files={"primary_image": ("scene.png", png_bytes(self.array), "image/png")},
            )
        result = response.json()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["grounding_result"]["detections"], [])
        self.assertIsNone(result["grounding_result"]["annotated_preview_url"])
        self.assertEqual(result["answer"], "No confident region found for 'stadium'.")

    def test_api_sar_is_explicitly_not_implemented_without_loading(self):
        grounder = RemoteSensingGrounder()
        response = self.query("Find the bridge.", modality="sar", grounder=grounder)
        result = response.json()
        self.assertEqual(result["status"], "not_implemented")
        self.assertIsNone(result["grounding_result"])
        self.assertEqual(result["result_status"], "UNSUPPORTED_INPUT")
        self.assertIn("incompatible", result["answer"])
        self.assertEqual(grounder.health().status, "unloaded")

    def test_unsupported_grounding_target_is_not_sent_to_model(self):
        grounder = RemoteSensingGrounder()
        response = self.query("Locate the clouds.", grounder=grounder)
        result = response.json()
        self.assertEqual(result["task"], "unsupported")
        self.assertEqual(result["status"], "not_implemented")
        self.assertEqual(grounder.health().status, "unloaded")

    def test_grounding_report_json_csv_pdf_and_zip_include_detections(self):
        result = self.query("Highlight the water bodies.").json()
        report = self.client.post("/api/agent/report", json={"request_id": result["request_id"], "formats": ["pdf", "json", "csv", "zip"]})
        self.assertEqual(report.status_code, 200, report.text)
        artifacts = {item["format"]: item for item in report.json()["artifacts"]}
        document = self.client.get(artifacts["json"]["url"]).json()
        self.assertEqual(document["user_query"]["target_concept"], "water body")
        self.assertEqual(document["statistics"]["grounding_statistics"]["detection_count"], 2)
        self.assertIn("Grounding DINO", " ".join(document["scientific_limitations"]))
        csv_text = self.client.get(artifacts["csv"]["url"]).text
        self.assertIn("grounding_detections", csv_text)
        self.assertIn("bbox_world", csv_text)
        pdf = self.client.get(artifacts["pdf"]["url"]).content
        self.assertTrue(pdf.startswith(b"%PDF"))
        package = zipfile.ZipFile(io.BytesIO(self.client.get(artifacts["zip"]["url"]).content))
        self.assertTrue(any(name.startswith("evidence/") and "Grounding" in name for name in package.namelist()))

    def test_health_exposes_grounder_without_forcing_load(self):
        grounder = RemoteSensingGrounder()
        with patch("satquery_agent.api.get_grounder", return_value=grounder):
            health = self.client.get("/api/agent/health").json()
        self.assertEqual(health["specialists"]["rs_grounder"]["status"], "unloaded")
        self.assertEqual(grounder.health().status, "unloaded")

    @unittest.skipUnless(os.getenv("SATQUERY_RUN_REAL_GROUNDER") == "1", "real multipart Grounding DINO API test is opt-in")
    def test_real_checkpoint_runs_through_multipart_api(self):
        sample_path = "satquery_agent/demo_samples/single-optical.png"
        with open(sample_path, "rb") as source:
            response = self.client.post(
                "/api/agent/query",
                data={"query": "Locate all buildings.", "input_mode": "single", "primary_modality": "optical", "use_cache": "false"},
                files={"primary_image": ("single-optical.png", source.read(), "image/png")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["grounding_result"]["detections"])
        self.assertTrue(result["grounding_result"]["annotated_preview_url"])
        self.assertEqual(result["model"]["checkpoint"], "IDEA-Research/grounding-dino-tiny")
        self.assertEqual(result["grounding_result"]["device"], "cpu")
        self.assertEqual(self.client.get(result["grounding_result"]["annotated_preview_url"]).status_code, 200)


if __name__ == "__main__":
    unittest.main()
