"""Strict verification for the SatQuery Vision Encoder v1 artifact bundle."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


ADAPTER_FILENAME = "satquery_vision_encoder_v1_adapter.pt"
EXPECTED_ADAPTER_SHA256 = "a99c0bf0fb44044988ef1698483888c8a2e3a047d2d2d56478837575cf7626ea"
EXPECTED_MODEL_VERSION = "1.0.0"
EXPECTED_BACKBONE = "OpenCLIP ViT-L-14"
EXPECTED_OPENCLIP_MODEL = "ViT-L-14"
EXPECTED_PRETRAINED = "laion2b_s32b_b82k"
EXPECTED_EMBEDDING_DIMENSION = 768
REQUIRED_FILES = (
    ADAPTER_FILENAME,
    "model_card.json",
    "preprocessing.json",
    "baseline_vs_adapted.json",
    "sha256.txt",
)
EXPECTED_GENERIC_METRICS = {
    "loss": 4.8016743659973145,
    "image_to_text_r1": 0.0021824531722813845,
    "text_to_image_r1": 0.0008729812107048929,
    "image_to_text_r5": 0.010039283894002438,
    "text_to_image_r5": 0.004364906344562769,
    "image_to_text_r10": 0.021388040855526924,
    "text_to_image_r10": 0.00654735928401351,
}
EXPECTED_ADAPTED_METRICS = {
    "loss": 4.120570659637451,
    "image_to_text_r1": 0.005674378015100956,
    "text_to_image_r1": 0.006110868416726589,
    "image_to_text_r5": 0.022697512060403824,
    "text_to_image_r5": 0.03491925075650215,
    "image_to_text_r10": 0.050632912665605545,
    "text_to_image_r10": 0.05805325135588646,
}


class SVEError(RuntimeError):
    """Base class for internal SVE failures with a safe public message."""

    public_message = "SatQuery Vision Encoder v1 is unavailable. Existing analysis remains available."


class SVEArtifactMissing(SVEError):
    public_message = "SatQuery Vision Encoder v1 artifacts are unavailable. Existing analysis remains available."


class SVEChecksumMismatch(SVEError):
    public_message = "SatQuery Vision Encoder v1 failed artifact verification. Existing analysis remains available."


class SVEInvalidMetadata(SVEError):
    public_message = "SatQuery Vision Encoder v1 metadata is invalid. Existing analysis remains available."


class SVEUnsupportedVersion(SVEError):
    public_message = "This SatQuery Vision Encoder artifact version is unsupported. Existing analysis remains available."


class SVELoadFailure(SVEError):
    public_message = "SatQuery Vision Encoder v1 could not be loaded. Existing analysis remains available."


class SVEInferenceFailure(SVEError):
    public_message = "SatQuery Vision Encoder v1 evidence could not be computed. Existing analysis remains available."


@dataclass(frozen=True)
class VerifiedSVEArtifacts:
    """Internal-only resolved artifacts. Paths must never be serialized publicly."""

    model_dir: Path
    adapter_path: Path
    adapter_sha256: str
    model_card: Dict[str, Any]
    preprocessing: Dict[str, Any]
    evaluation: Dict[str, Any]

    @property
    def checksum_fingerprint(self) -> str:
        return f"sha256:{self.adapter_sha256[:12]}"


def configured_model_dir() -> Path:
    configured = os.getenv("SVE_MODEL_DIR", "models/satquery_vision_encoder_v1").strip()
    if not configured:
        raise SVEArtifactMissing("SVE_MODEL_DIR is empty")
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[1] / candidate
    return candidate.resolve()


def _json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SVEInvalidMetadata(f"Invalid metadata file: {path.name}") from error
    if not isinstance(value, dict):
        raise SVEInvalidMetadata(f"Metadata must be an object: {path.name}")
    return value


def _require_equal(payload: Dict[str, Any], key: str, expected: Any, filename: str) -> None:
    if payload.get(key) != expected:
        raise SVEInvalidMetadata(f"Unexpected {key} in {filename}")


def _validate_metrics(evaluation: Dict[str, Any], model_card: Dict[str, Any]) -> None:
    required = {
        "loss", "image_to_text_r1", "text_to_image_r1", "image_to_text_r5",
        "text_to_image_r5", "image_to_text_r10", "text_to_image_r10",
    }
    generic = evaluation.get("generic_openclip")
    adapted = evaluation.get("satquery_vision_encoder_v1")
    if not isinstance(generic, dict) or not isinstance(adapted, dict):
        raise SVEInvalidMetadata("Validation comparison is missing model sections")
    if set(generic) != required or set(adapted) != required:
        raise SVEInvalidMetadata("Validation comparison contains unexpected metric keys")
    if any(not isinstance(value, (int, float)) for value in [*generic.values(), *adapted.values()]):
        raise SVEInvalidMetadata("Validation metrics must be numeric")
    if model_card.get("validation_metrics") != adapted or model_card.get("generic_openclip_baseline") != generic:
        raise SVEInvalidMetadata("Model card metrics do not match the validation comparison")
    if generic != EXPECTED_GENERIC_METRICS or adapted != EXPECTED_ADAPTED_METRICS:
        raise SVEInvalidMetadata("Validation metrics do not match the verified release")


def verify_sve_artifacts(model_dir: Path | None = None) -> VerifiedSVEArtifacts:
    directory = (model_dir or configured_model_dir()).resolve()
    if not directory.is_dir():
        raise SVEArtifactMissing("Configured model directory does not exist")
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise SVEArtifactMissing("Required artifact files are missing")

    try:
        checksum_parts = (directory / "sha256.txt").read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as error:
        raise SVEInvalidMetadata("Checksum manifest could not be read") from error
    if len(checksum_parts) != 2 or checksum_parts[1].lstrip("*") != ADAPTER_FILENAME:
        raise SVEInvalidMetadata("Checksum manifest does not identify the expected adapter")
    declared = checksum_parts[0].lower()
    if declared != EXPECTED_ADAPTER_SHA256:
        raise SVEChecksumMismatch("Checksum manifest does not match the verified release")

    adapter_path = directory / ADAPTER_FILENAME
    digest = hashlib.sha256()
    try:
        with adapter_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SVEArtifactMissing("Adapter could not be read") from error
    actual = digest.hexdigest()
    if actual != EXPECTED_ADAPTER_SHA256:
        raise SVEChecksumMismatch("Adapter checksum does not match the verified release")

    model_card = _json_object(directory / "model_card.json")
    if model_card.get("version") != EXPECTED_MODEL_VERSION:
        raise SVEUnsupportedVersion("Unsupported model version")
    _require_equal(model_card, "name", "SatQuery Vision Encoder v1", "model_card.json")
    _require_equal(model_card, "backbone", EXPECTED_BACKBONE, "model_card.json")
    _require_equal(model_card, "pretrained_weights", EXPECTED_PRETRAINED, "model_card.json")
    _require_equal(model_card, "embedding_dimension", EXPECTED_EMBEDDING_DIMENSION, "model_card.json")
    _require_equal(model_card, "adaptation_dataset", "BigEarthNet.txt", "model_card.json")
    _require_equal(model_card, "image_source", "BigEarthNet v2 Lithuania Summer", "model_card.json")
    _require_equal(model_card, "training_pairs", 4008, "model_card.json")
    _require_equal(model_card, "validation_pairs", 2291, "model_card.json")
    _require_equal(model_card, "test_pairs", 2053, "model_card.json")
    if (model_card.get("input") or {}).get("mode") != "RGB" or (model_card.get("input") or {}).get("input_size") != [224, 224]:
        raise SVEInvalidMetadata("Model card input contract is invalid")

    preprocessing = _json_object(directory / "preprocessing.json")
    _require_equal(preprocessing, "model_name", EXPECTED_OPENCLIP_MODEL, "preprocessing.json")
    _require_equal(preprocessing, "pretrained", EXPECTED_PRETRAINED, "preprocessing.json")
    _require_equal(preprocessing, "input_size", 224, "preprocessing.json")
    _require_equal(preprocessing, "color_mode", "RGB", "preprocessing.json")
    _require_equal(preprocessing, "embedding_normalization", "L2", "preprocessing.json")
    _require_equal(preprocessing, "openclip_transform", True, "preprocessing.json")
    _require_equal(preprocessing, "sentinel_2_band_order", ["B04", "B03", "B02"], "preprocessing.json")
    _require_equal(preprocessing, "reflectance_scale", 10000.0, "preprocessing.json")

    evaluation = _json_object(directory / "baseline_vs_adapted.json")
    _validate_metrics(evaluation, model_card)
    return VerifiedSVEArtifacts(
        model_dir=directory,
        adapter_path=adapter_path,
        adapter_sha256=actual,
        model_card=model_card,
        preprocessing=preprocessing,
        evaluation=evaluation,
    )
