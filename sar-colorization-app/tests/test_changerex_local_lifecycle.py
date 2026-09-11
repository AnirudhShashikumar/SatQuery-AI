from pathlib import Path

import pytest
import torch

import changerex_local.lifecycle as lifecycle_module
from changerex_local.lifecycle import (
    ChangerExLifecycle,
    DeviceSelectionError,
    LifecycleError,
    configure_lifecycle,
    select_device,
)
from changerex_local.schemas import CheckpointVerificationReport


def _report(path):
    return CheckpointVerificationReport(
        path=str(path), file_size=1, sha256="a" * 64, top_level_keys=("state_dict",),
        tensor_count=1, parameter_count=1, missing_keys=(), unexpected_keys=(),
        key_transformations=(), checkpoint_metadata={},
    )


def test_device_selection_cpu_mps_and_explicit_fallback(monkeypatch):
    assert select_device("cpu") == ("cpu", [])
    monkeypatch.setattr(lifecycle_module, "_mps_is_verified", lambda: True)
    assert select_device("auto")[0] == "mps"
    assert select_device("mps")[0] == "mps"
    monkeypatch.setattr(lifecycle_module, "_mps_is_verified", lambda: False)
    with pytest.raises(DeviceSelectionError, match="MPS was requested"):
        select_device("mps")
    selected, warnings = select_device("mps", allow_fallback=True)
    assert selected == "cpu" and warnings


def test_lazy_load_singleton_reuse_and_load_once(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x")
    model = torch.nn.Linear(1, 1, bias=False)
    monkeypatch.setattr(lifecycle_module, "build_model", lambda: model)
    monkeypatch.setattr(lifecycle_module, "load_checkpoint_strict", lambda m, p: _report(p))
    monkeypatch.setattr(lifecycle_module, "PARAMETER_COUNT", 1)
    instance = ChangerExLifecycle(checkpoint, device="cpu")
    assert instance.state == "unloaded" and instance.load_count == 0
    first, reused = instance.ensure_loaded()
    second, reused_second = instance.ensure_loaded()
    assert first is second is model
    assert not reused and reused_second
    assert instance.load_count == 1 and instance.reuse_count == 1


def test_sticky_load_failure_and_singleton_immutability(monkeypatch, tmp_path):
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"bad")
    monkeypatch.setattr(lifecycle_module, "load_checkpoint_strict", lambda *args: (_ for _ in ()).throw(ValueError("secret\nload failure")))
    instance = ChangerExLifecycle(checkpoint, device="cpu")
    with pytest.raises(LifecycleError, match="load failure"):
        instance.ensure_loaded()
    with pytest.raises(LifecycleError, match="sticky failed"):
        instance.ensure_loaded()
    assert instance.state == "failed" and "\n" not in instance.safe_error

    lifecycle_module._reset_lifecycle_for_tests()
    configure_lifecycle(checkpoint, device="cpu")
    with pytest.raises(LifecycleError, match="already configured"):
        configure_lifecycle(tmp_path / "other.pt", device="cpu")
    lifecycle_module._reset_lifecycle_for_tests()
