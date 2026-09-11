"""Thread-safe, lazy, single-model lifecycle for ChangerEx."""

from __future__ import annotations

import os
import platform
import resource
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .architecture import ChangerEx, build_model, model_parameter_count
from .checkpoint import load_checkpoint_strict
from .config import PARAMETER_COUNT
from .schemas import CheckpointVerificationReport


class DeviceSelectionError(RuntimeError):
    pass


class LifecycleError(RuntimeError):
    pass


def _mps_is_verified() -> bool:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        return False
    try:
        test = torch.ones(1, device="mps") + 1
        torch.mps.synchronize()
        return float(test.cpu().item()) == 2.0
    except Exception:
        return False


def select_device(requested: str, *, allow_fallback: bool = False) -> tuple[str, list[str]]:
    requested = requested.lower().strip()
    if requested not in {"auto", "mps", "cpu"}:
        raise DeviceSelectionError("device must be one of: auto, mps, cpu")
    warnings: list[str] = []
    if requested == "cpu":
        return "cpu", warnings
    if _mps_is_verified():
        return "mps", warnings
    if requested == "auto":
        warnings.append("MPS was unavailable or failed verification; auto selected CPU.")
        return "cpu", warnings
    if allow_fallback:
        warnings.append("Explicit MPS request could not be satisfied; CPU fallback was explicitly allowed.")
        return "cpu", warnings
    raise DeviceSelectionError(
        "MPS was requested but is unavailable or failed verification. "
        "Use --allow-device-fallback to permit CPU fallback."
    )


def peak_memory_mb(device: str | None = None) -> float | None:
    try:
        if device == "mps" and hasattr(torch, "mps"):
            return float(torch.mps.driver_allocated_memory() / (1024 * 1024))
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux reports KiB.
        return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024
    except Exception:
        return None


class ChangerExLifecycle:
    """A sticky lifecycle. Failed instances are not retried in-process."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        allow_device_fallback: bool = False,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.requested_device = device
        self.allow_device_fallback = allow_device_fallback
        self.state = "unloaded"
        self.selected_device: str | None = None
        self.checkpoint_verified = False
        self.load_count = 0
        self.reuse_count = 0
        self.inference_count = 0
        self.load_time = 0.0
        self.last_inference_time = 0.0
        self.safe_error: str | None = None
        self.model_parameter_count = 0
        self.warnings: list[str] = []
        self.checkpoint_report: CheckpointVerificationReport | None = None
        self._model: ChangerEx | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _safe_error(error: BaseException) -> str:
        message = f"{type(error).__name__}: {error}".replace("\n", " ")
        return message[:500]

    def ensure_loaded(self) -> tuple[ChangerEx, bool]:
        with self._lock:
            if self.state == "ready":
                self.reuse_count += 1
                assert self._model is not None
                return self._model, True
            if self.state == "failed":
                raise LifecycleError(f"ChangerEx lifecycle is in sticky failed state: {self.safe_error}")
            if self.state == "loading":
                raise LifecycleError("Reentrant ChangerEx loading was detected")
            self.state = "loading"
            started = time.perf_counter()
            try:
                selected, warnings = select_device(
                    self.requested_device, allow_fallback=self.allow_device_fallback
                )
                model = build_model()
                report = load_checkpoint_strict(model, self.checkpoint_path)
                if model_parameter_count(model) != PARAMETER_COUNT:
                    raise LifecycleError(
                        f"Architecture parameter count mismatch: expected {PARAMETER_COUNT}, "
                        f"received {model_parameter_count(model)}"
                    )
                model.eval().to(device=selected, dtype=torch.float32)
                self.selected_device = selected
                self.warnings.extend(warnings)
                self.checkpoint_report = report
                self.checkpoint_verified = True
                self.model_parameter_count = model_parameter_count(model)
                self._model = model
                self.load_count += 1
                self.state = "ready"
                self.load_time = time.perf_counter() - started
                return model, False
            except Exception as error:
                self.state = "failed"
                self.safe_error = self._safe_error(error)
                self.load_time = time.perf_counter() - started
                self._model = None
                raise LifecycleError(self.safe_error) from error

    def record_inference(self, elapsed: float) -> None:
        with self._lock:
            self.inference_count += 1
            self.last_inference_time = elapsed

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "requested_device": self.requested_device,
                "selected_device": self.selected_device,
                "checkpoint_verified": self.checkpoint_verified,
                "load_count": self.load_count,
                "reuse_count": self.reuse_count,
                "inference_count": self.inference_count,
                "load_time": self.load_time,
                "last_inference_time": self.last_inference_time,
                "safe_error": self.safe_error,
                "model_parameter_count": self.model_parameter_count,
                "peak_memory_mb": peak_memory_mb(self.selected_device),
                "warnings": list(self.warnings),
                "checkpoint_report": asdict(self.checkpoint_report) if self.checkpoint_report else None,
            }


_singleton_lock = threading.RLock()
_singleton: ChangerExLifecycle | None = None


def configure_lifecycle(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    allow_device_fallback: bool = False,
) -> ChangerExLifecycle:
    """Configure the one process-wide lifecycle without loading it yet."""
    global _singleton
    resolved = str(Path(checkpoint_path).expanduser().resolve())
    with _singleton_lock:
        if _singleton is None:
            _singleton = ChangerExLifecycle(
                resolved, device=device, allow_device_fallback=allow_device_fallback
            )
            return _singleton
        if (
            _singleton.checkpoint_path != resolved
            or _singleton.requested_device != device
            or _singleton.allow_device_fallback != allow_device_fallback
        ):
            raise LifecycleError(
                "ChangerEx is already configured in this process; checkpoint and device are immutable"
            )
        return _singleton


def get_lifecycle(
    *, device: str = "auto", allow_device_fallback: bool = False
) -> ChangerExLifecycle:
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            if device != "auto" and _singleton.requested_device not in {"auto", device}:
                raise LifecycleError(
                    f"ChangerEx is configured for {_singleton.requested_device}, not {device}"
                )
            return _singleton
        checkpoint = os.environ.get("CHANGEREX_CHECKPOINT")
        if not checkpoint:
            raise LifecycleError(
                "ChangerEx is not configured. Call configure_lifecycle(checkpoint_path, ...) "
                "or set CHANGEREX_CHECKPOINT."
            )
        _singleton = ChangerExLifecycle(
            checkpoint, device=device, allow_device_fallback=allow_device_fallback
        )
        return _singleton


def _reset_lifecycle_for_tests() -> None:
    """Test-only reset; production callers must treat failure as sticky."""
    global _singleton
    with _singleton_lock:
        _singleton = None
