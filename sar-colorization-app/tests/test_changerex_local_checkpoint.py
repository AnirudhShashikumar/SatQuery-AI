import os
from pathlib import Path

import pytest
import torch

from changerex_local.architecture import build_model
from changerex_local.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointVerificationError,
    inspect_checkpoint,
    load_checkpoint_strict,
    sha256_file,
)
from changerex_local.config import CHECKPOINT_SHA256, PARAMETER_COUNT


def _save(path: Path, state_dict):
    torch.save({"state_dict": state_dict, "meta": {"source": "test"}}, path)
    return sha256_file(path)


def test_checkpoint_sha_verification_and_invalid_rejection(tmp_path):
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "valid.pt"
    digest = _save(path, model.state_dict())
    state, report = inspect_checkpoint(path, expected_sha256=digest)
    assert set(state) == {"weight", "bias"}
    assert report.sha256 == digest
    assert report.top_level_keys == ("meta", "state_dict")
    with pytest.raises(CheckpointVerificationError, match="SHA-256 mismatch"):
        inspect_checkpoint(path, expected_sha256="0" * 64)


def test_strict_loading_missing_and_unexpected_keys(tmp_path):
    model = torch.nn.Linear(2, 1)
    missing = tmp_path / "missing.pt"
    digest = _save(missing, {"weight": model.weight.detach().clone()})
    with pytest.raises(CheckpointCompatibilityError, match="Strict state-dict"):
        load_checkpoint_strict(model, missing, expected_sha256=digest)

    unexpected = tmp_path / "unexpected.pt"
    state = dict(model.state_dict())
    state["extra"] = torch.zeros(1)
    digest = _save(unexpected, state)
    with pytest.raises(CheckpointCompatibilityError, match="Strict state-dict"):
        load_checkpoint_strict(model, unexpected, expected_sha256=digest)


@pytest.mark.skipif(
    not os.environ.get("CHANGEREX_TEST_CHECKPOINT"),
    reason="set CHANGEREX_TEST_CHECKPOINT for the real-checkpoint integration test",
)
def test_real_official_checkpoint_strict_loads():
    checkpoint = os.environ["CHANGEREX_TEST_CHECKPOINT"]
    model = build_model()
    report = load_checkpoint_strict(model, checkpoint)
    assert report.sha256 == CHECKPOINT_SHA256
    assert report.parameter_count == PARAMETER_COUNT
    assert report.tensor_count == 173
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert report.key_transformations == ()
