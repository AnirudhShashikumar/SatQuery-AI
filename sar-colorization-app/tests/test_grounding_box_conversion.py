"""Focused regression tests for Grounding DINO box-coordinate handling."""

from __future__ import annotations

import ast
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from satquery_agent.image_ingestion import remove_preview_url
from satquery_agent.models import ImageFormat, ImageMetadata, Modality
from satquery_agent.specialists.grounder import (
    POSTPROCESSED_BOX_FORMAT,
    RAW_BOX_FORMAT,
    RemoteSensingGrounder,
    _audit_normalized_cxcywh_to_source_xyxy,
    _clip_source_xyxy,
    _grounding_target_sizes,
    _raw_box_is_full_image,
    evaluate_detection_quality,
    sanitize_detections,
)


def metadata(width: int, height: int) -> ImageMetadata:
    return ImageMetadata(
        file_id="box-conversion-test",
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


class ContractModel:
    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **_kwargs):
        return SimpleNamespace(
            pred_boxes=torch.tensor([[[0.5, 0.5, 0.25, 0.5]]], dtype=torch.float32),
            logits=torch.tensor([[[2.0]]], dtype=torch.float32),
        )


class ContractProcessor:
    def __call__(self, *, images, text, return_tensors):
        self.source_size = images.size
        return {
            "pixel_values": torch.zeros((1, 3, 320, 640), dtype=torch.float32),
            "input_ids": torch.ones((1, 4), dtype=torch.long),
        }

    def post_process_grounded_object_detection(
        self, outputs, input_ids, *, box_threshold, text_threshold, target_sizes
    ):
        self.target_sizes = target_sizes.detach().clone()
        height, width = (int(value) for value in target_sizes[0].tolist())
        box = _audit_normalized_cxcywh_to_source_xyxy(outputs.pred_boxes[0, 0], width, height)
        return [{"boxes": torch.tensor([box]), "scores": torch.tensor([0.75]), "labels": ["water body"]}]


class GroundingBoxConversionTests(unittest.TestCase):
    def test_square_image_conversion(self):
        result = _audit_normalized_cxcywh_to_source_xyxy([0.5, 0.5, 0.5, 0.5], 256, 256)
        self.assertEqual(result, [64.0, 64.0, 192.0, 192.0])

    def test_non_square_image_conversion(self):
        result = _audit_normalized_cxcywh_to_source_xyxy([0.5, 0.5, 0.5, 0.5], 640, 320)
        self.assertEqual(result, [160.0, 80.0, 480.0, 240.0])

    def test_target_size_uses_height_then_width(self):
        target_sizes = _grounding_target_sizes(source_width=640, source_height=320)
        self.assertTrue(torch.equal(target_sizes, torch.tensor([[320, 640]], dtype=torch.int64)))

    def test_known_cxcywh_to_xyxy_example(self):
        result = _audit_normalized_cxcywh_to_source_xyxy([0.5, 0.5, 0.25, 0.5], 400, 200)
        self.assertEqual(result, [150.0, 50.0, 250.0, 150.0])

    def test_postprocessed_pixels_are_not_scaled_again(self):
        result = sanitize_detections(
            [[150.0, 50.0, 250.0, 150.0]],
            [0.75],
            ["water body"],
            metadata(400, 200),
            "water body",
        )
        self.assertEqual(result[0].bbox_pixels, [150, 50, 250, 150])

    def test_processor_restores_non_square_source_coordinates(self):
        grounder = RemoteSensingGrounder()
        processor = ContractProcessor()
        image = Image.new("RGB", (640, 320))
        with patch.object(grounder, "_load_components", return_value=(processor, ContractModel())):
            result = grounder.ground(
                image,
                "Highlight the water body.",
                metadata(640, 320),
                Modality.OPTICAL,
                ["r", "g", "b"],
                "RGB",
            )
        self.assertEqual(processor.target_sizes.tolist(), [[320, 640]])
        self.assertEqual(result.detections[0].bbox_pixels, [240, 80, 400, 240])
        self.assertEqual(processor.source_size, (640, 320))
        remove_preview_url(result.annotated_preview_url)
        image.close()

    def test_clipping_at_image_boundaries_records_reason(self):
        audit = []
        result = sanitize_detections(
            [[-10.0, -4.0, 410.0, 205.0]],
            [0.8],
            ["vegetation"],
            metadata(400, 200),
            "vegetation",
            audit_records=audit,
        )
        self.assertEqual(result[0].bbox_pixels, [0, 0, 400, 200])
        self.assertTrue(audit[0]["clipping_occurred"])
        self.assertEqual(audit[0]["clipping_reason"], "postprocessed_box_outside_source_bounds")
        self.assertEqual(_clip_source_xyxy([-10, -4, 410, 205], 400, 200)[0], [0, 0, 400, 200])

    def test_raw_full_image_box_remains_identifiable_as_model_output(self):
        raw = [0.50008339, 0.50020701, 1.0, 0.99999952]
        postprocessed = _audit_normalized_cxcywh_to_source_xyxy(raw, 256, 256)
        audit = []
        result, rejected = evaluate_detection_quality(
            [postprocessed], [0.37422514], ["water body"], metadata(256, 256), "water body", audit_records=audit
        )
        self.assertTrue(_raw_box_is_full_image(raw))
        self.assertEqual(result, [])
        self.assertEqual(rejected[0].bbox_pixels, [0, 0, 256, 256])
        self.assertFalse(audit[0]["clipping_occurred"])

    def test_small_raw_box_does_not_become_full_image(self):
        raw = [0.5, 0.5, 0.1, 0.2]
        postprocessed = _audit_normalized_cxcywh_to_source_xyxy(raw, 256, 256)
        result = sanitize_detections(
            [postprocessed], [0.8], ["water body"], metadata(256, 256), "water body"
        )
        self.assertFalse(_raw_box_is_full_image(raw))
        self.assertEqual(result[0].bbox_pixels, [115, 102, 141, 154])

    def test_debug_diagnostic_is_bounded_and_records_box_contract(self):
        grounder = RemoteSensingGrounder()
        processor = ContractProcessor()
        image = Image.new("RGB", (400, 200))
        with self.assertLogs("satquery_agent.specialists.grounder", level="DEBUG") as logs:
            with patch.object(grounder, "_load_components", return_value=(processor, ContractModel())):
                result = grounder.ground(
                    image,
                    "Highlight the water body.",
                    metadata(400, 200),
                    Modality.OPTICAL,
                    ["r", "g", "b"],
                    "RGB",
                )
        payload = ast.literal_eval(logs.output[-1].split("Grounding DINO box audit: ", 1)[1])
        self.assertEqual(payload["raw_box_format"], RAW_BOX_FORMAT)
        self.assertEqual(payload["postprocessed_box_format"], POSTPROCESSED_BOX_FORMAT)
        self.assertEqual(payload["target_sizes"], [[200, 400]])
        self.assertEqual(payload["processor_input_dimensions"], {"height": 320, "width": 640})
        self.assertFalse(payload["manual_scaling_occurred"])
        final = payload["final_box_values"][0]
        self.assertEqual(final["score"], 0.75)
        self.assertEqual(final["label"], "water body")
        self.assertIn("clipping_occurred", final)
        remove_preview_url(result.annotated_preview_url)
        image.close()


if __name__ == "__main__":
    unittest.main()
