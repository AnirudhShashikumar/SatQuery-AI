"""Production adapter for the validated standalone Pure-PyTorch ChangerEx.

The adapter owns no model. It delegates loading and inference to the standalone
package's immutable process-wide lifecycle, and translates failures into a
small, path-free contract so the deterministic detector can always take over.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from changerex_local import ChangerExInferenceError, configure_lifecycle, predict_change
from changerex_local.config import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    DEFAULT_MAXIMUM_DIMENSION,
    DEFAULT_THRESHOLD,
    MODEL_NAME,
)
from changerex_local.lifecycle import LifecycleError, get_lifecycle


MODEL_ID = MODEL_NAME
ARCHITECTURE = "ChangerEx + IA-ResNetV1c-18 backbone"
TRAINING_DATASET = "LEVIR-CD"
CHECKPOINT_NAME = CHECKPOINT_FILENAME
CHECKPOINT_FINGERPRINT = CHECKPOINT_SHA256[:12]
DOMAIN_WARNING = (
    "ChangerEx was trained on LEVIR-CD building-change imagery. Generalization may be weaker for "
    "vegetation change, flooding, seasonal variation, unfamiliar sensors, SAR imagery or non-urban scenes."
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = _REPOSITORY_ROOT / "models" / "changerex" / CHECKPOINT_FILENAME
LOGGER = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("SATQUERY_CHANGEREX_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _threshold() -> float:
    try:
        value = float(os.getenv("SATQUERY_CHANGEREX_THRESHOLD", str(DEFAULT_THRESHOLD)))
    except ValueError:
        return DEFAULT_THRESHOLD
    return value if 0.0 <= value <= 1.0 else DEFAULT_THRESHOLD


def selected_change_engine() -> str:
    """Select the internal engine while preserving explicit legacy TTP settings."""
    explicit = os.getenv("SATQUERY_CHANGE_ENGINE")
    if explicit is not None:
        selected = explicit.strip().lower()
        return selected if selected in {"changerex", "ttp", "deterministic"} else "changerex"
    # Existing deployments that explicitly configured TTP keep that behavior.
    if "TTP_DEFAULT_MODE" in os.environ:
        legacy = os.environ["TTP_DEFAULT_MODE"].strip().lower()
        return "deterministic" if legacy == "deterministic" else "ttp"
    # A disabled local engine preserves the legacy TTP-selection behavior. TTP
    # may itself be disabled, in which case the established deterministic path
    # remains byte-for-byte equivalent.
    if not _enabled():
        return "ttp"
    return "changerex"


class ChangerExProductionError(RuntimeError):
    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


@dataclass
class ChangerExClientResult:
    mask: np.ndarray
    changed_percentage: float
    changed_pixels: int
    region_count: int
    largest_region_pixels: int
    runtime_ms: int
    model_load_ms: int
    reused_model: bool
    device: str
    threshold: float
    stage_durations_ms: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


class ChangerExChangeEngine:
    """Thin, lazy bridge to the standalone singleton lifecycle."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._configured = False
        self._last_error: str | None = None

    @property
    def checkpoint_path(self) -> Path:
        configured = os.getenv("SATQUERY_CHANGEREX_CHECKPOINT", "").strip()
        if not configured:
            return _DEFAULT_CHECKPOINT
        candidate = Path(configured).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (_REPOSITORY_ROOT / candidate).resolve()

    @property
    def device(self) -> str:
        value = os.getenv("SATQUERY_CHANGEREX_DEVICE", "auto").strip().lower()
        return value if value in {"auto", "mps", "cpu"} else "auto"

    def _configure(self) -> None:
        if not _enabled():
            raise ChangerExProductionError(
                "DISABLED", "ChangerEx is disabled; the deterministic change analyzer produced the result."
            )
        checkpoint = self.checkpoint_path
        if not checkpoint.is_file():
            LOGGER.warning("ChangerEx checkpoint is unavailable; deterministic fallback remains active.")
            raise ChangerExProductionError(
                "CHECKPOINT_UNAVAILABLE",
                "ChangerEx checkpoint is unavailable; the deterministic change analyzer produced the result.",
            )
        with self._lock:
            if not self._configured:
                try:
                    configure_lifecycle(checkpoint, device=self.device, allow_device_fallback=False)
                except LifecycleError as error:
                    self._last_error = str(error)[:500]
                    LOGGER.warning("ChangerEx initialization failed; deterministic fallback remains active: %s", type(error).__name__)
                    raise ChangerExProductionError(
                        "MODEL_LOAD_FAILED",
                        "ChangerEx could not be initialized; the deterministic change analyzer produced the result.",
                    ) from error
                self._configured = True

    def predict(self, earlier: Image.Image, later: Image.Image) -> ChangerExClientResult:
        self._configure()
        try:
            result = predict_change(
                earlier,
                later,
                device=self.device,
                threshold=_threshold(),
                maximum_dimension=_positive_int(
                    "SATQUERY_CHANGEREX_MAXIMUM_DIMENSION", DEFAULT_MAXIMUM_DIMENSION
                ),
                allow_device_fallback=False,
            )
        except (ChangerExInferenceError, LifecycleError, ValueError) as error:
            self._last_error = str(error)[:500]
            LOGGER.warning("ChangerEx inference failed; deterministic fallback remains active: %s", type(error).__name__)
            raise ChangerExProductionError(
                "INFERENCE_FAILED",
                "ChangerEx inference failed; the deterministic change analyzer produced the result.",
            ) from error
        self._last_error = None
        runtime = result.runtime
        reuse = result.load_reuse_status
        return ChangerExClientResult(
            mask=result.binary_mask.astype(bool, copy=False),
            changed_percentage=result.changed_percentage,
            changed_pixels=result.changed_pixel_count,
            region_count=result.connected_component_count,
            largest_region_pixels=result.largest_component_size,
            runtime_ms=max(0, round((runtime.inference_seconds if runtime else 0.0) * 1000)),
            model_load_ms=max(0, round((runtime.load_seconds if runtime else 0.0) * 1000)),
            reused_model=bool(reuse.get("was_reused", False)),
            device=result.selected_device,
            threshold=result.threshold,
            stage_durations_ms=dict(result.stage_durations_ms),
            warnings=list(result.warnings),
            limitations=list(result.limitations),
        )

    def health_payload(self) -> dict[str, Any]:
        """Return a path-free status without forcing model initialization."""
        if not _enabled():
            return self._health("disabled", loaded=False)
        if not self.checkpoint_path.is_file():
            return self._health(
                "failed",
                loaded=False,
                errors=["ChangerEx checkpoint is unavailable."],
            )
        try:
            lifecycle = get_lifecycle(device=self.device, allow_device_fallback=False)
        except LifecycleError:
            # A checkpoint can be available before the lazy lifecycle is configured.
            return self._health("unloaded", loaded=False)
        status = lifecycle.status()
        state = str(status.get("state", "unloaded"))
        errors = [str(status["safe_error"])] if status.get("safe_error") else []
        return self._health(
            state,
            loaded=state == "ready",
            device=status.get("selected_device"),
            load_count=int(status.get("load_count", 0)),
            reuse_count=int(status.get("reuse_count", 0)),
            inference_count=int(status.get("inference_count", 0)),
            errors=errors,
        )

    @staticmethod
    def _health(
        status: str,
        *,
        loaded: bool,
        device: str | None = None,
        load_count: int = 0,
        reuse_count: int = 0,
        inference_count: int = 0,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "lifecycle": status,
            "model": MODEL_ID,
            "device": device,
            "checkpoint": CHECKPOINT_NAME,
            "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
            "loaded": loaded,
            "load_count": load_count,
            "reuse_count": reuse_count,
            "inference_count": inference_count,
            "errors": errors or [],
        }


CHANGEREX_ENGINE = ChangerExChangeEngine()


def changerex_enabled() -> bool:
    return _enabled()
