from __future__ import annotations

import json
from pathlib import Path

import pytest

from satquery_agent.sve_artifacts import (
    SVEArtifactMissing,
    SVEChecksumMismatch,
    SVEInvalidMetadata,
    SVEUnsupportedVersion,
    verify_sve_artifacts,
)


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "satquery_vision_encoder_v1"


def bundle(tmp_path: Path) -> Path:
    target = tmp_path / "model"
    target.mkdir()
    for source in MODEL_DIR.iterdir():
        if source.name.endswith(".pt"):
            (target / source.name).symlink_to(source)
        else:
            (target / source.name).write_bytes(source.read_bytes())
    return target


def test_valid_verified_bundle() -> None:
    result = verify_sve_artifacts(MODEL_DIR)
    assert result.adapter_sha256 == "a99c0bf0fb44044988ef1698483888c8a2e3a047d2d2d56478837575cf7626ea"
    assert result.checksum_fingerprint == "sha256:a99c0bf0fb44"


def test_missing_required_file(tmp_path: Path) -> None:
    target = bundle(tmp_path)
    (target / "preprocessing.json").unlink()
    with pytest.raises(SVEArtifactMissing):
        verify_sve_artifacts(target)


def test_checksum_mismatch(tmp_path: Path) -> None:
    target = bundle(tmp_path)
    (target / "sha256.txt").write_text("0" * 64 + "  satquery_vision_encoder_v1_adapter.pt\n")
    with pytest.raises(SVEChecksumMismatch):
        verify_sve_artifacts(target)


def test_malformed_metadata(tmp_path: Path) -> None:
    target = bundle(tmp_path)
    (target / "preprocessing.json").write_text("{")
    with pytest.raises(SVEInvalidMetadata):
        verify_sve_artifacts(target)


def test_unsupported_version(tmp_path: Path) -> None:
    target = bundle(tmp_path)
    card = json.loads((target / "model_card.json").read_text())
    card["version"] = "2.0.0"
    (target / "model_card.json").write_text(json.dumps(card))
    with pytest.raises(SVEUnsupportedVersion):
        verify_sve_artifacts(target)
