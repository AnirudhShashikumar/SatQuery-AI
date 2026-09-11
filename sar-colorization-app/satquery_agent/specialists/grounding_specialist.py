"""Lazy loader for the VRSBench-trained Grounding Specialist v1.1 scoring head."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

from ..models import SpecialistHealth


logger = logging.getLogger(__name__)

MODEL_NAME = "SatQuery Grounding Specialist v1.1"
CHECKPOINT_FILENAME = "grounding_specialist_v1_1_head.pt"
BASE_MODEL = "IDEA-Research/grounding-dino-tiny"
TRAINING_DATASET = "VRSBench"
TRAINING_RECIPE_VERSION = "v1.1"
CHECKPOINT_STEP = 600
EXPECTED_CHECKPOINT_SHA256 = "5e8db30becadb1d063fc0154614ee2a2a4b7a2c2923db3fce4ff036c65007342"


class GroundingSpecialistError(RuntimeError):
    """Safe internal loading or scoring failure."""


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def configured_model_dir() -> Path:
    configured = os.getenv("SATQUERY_GROUNDING_SPECIALIST_MODEL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "models" / "grounding_specialist_v1_1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GroundingSpecialistError(f"{path.name} must contain a JSON object")
    return value


def _build_head(torch: Any) -> Any:
    nn = torch.nn

    class GroundingSpecialistHead(nn.Module):
        """Exact architecture exported by Grounding Specialist v1.1 training."""

        def __init__(self) -> None:
            super().__init__()
            self.max_box_delta = 0.15
            self.query_scorer = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
            )
            self.box_refiner = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 4),
                nn.Tanh(),
            )

        def forward(self, hidden_state: Any, base_boxes_cxcywh: Any) -> tuple[Any, Any]:
            query_logits = self.query_scorer(hidden_state).squeeze(-1)
            refined = base_boxes_cxcywh + self.box_refiner(hidden_state) * self.max_box_delta
            centers = refined[..., :2].clamp(0.0, 1.0)
            sizes = refined[..., 2:].clamp(1e-4, 1.0)
            return query_logits, torch.cat((centers, sizes), dim=-1)

    return GroundingSpecialistHead()


class GroundingSpecialist:
    def __init__(self, *, model_dir: Optional[Path] = None, enabled: Optional[bool] = None) -> None:
        self.model_dir = (model_dir or configured_model_dir()).resolve()
        self._enabled = _bool_env("SATQUERY_GROUNDING_SPECIALIST_ENABLED", True) if enabled is None else enabled
        self._load_lock = threading.RLock()
        self._head: Any = None
        self._device: Optional[str] = None
        self._state = "unloaded" if self._enabled else "disabled"
        self._error: Optional[str] = None
        self._load_count = 0
        self.checkpoint_hash: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._state == "ready" and self._head is not None

    @property
    def load_count(self) -> int:
        return self._load_count

    def health(self) -> SpecialistHealth:
        return SpecialistHealth(
            status=self._state,
            device=self._device,
            error=self._error,
            model_id=f"{MODEL_NAME} step {CHECKPOINT_STEP}",
            load_source="verified_local_bundle" if self.loaded else None,
            last_error=self._error,
            smoke_verified=self.loaded,
        )

    def load(self, device: str) -> None:
        if self.loaded and self._device == device:
            return
        if not self._enabled:
            self._state = "disabled"
            raise GroundingSpecialistError("Grounding Specialist v1.1 is disabled")
        with self._load_lock:
            if self.loaded:
                if self._device != device:
                    self._head.to(device)
                    self._device = device
                return
            if self._state == "failed":
                raise GroundingSpecialistError("Grounding Specialist v1.1 previously failed to load")
            self._state = "loading"
            self._error = None
            try:
                import torch

                checkpoint_path = self.model_dir / CHECKPOINT_FILENAME
                checkpoint_hash = _sha256(checkpoint_path)
                if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
                    raise GroundingSpecialistError("Grounding specialist checkpoint hash mismatch")
                manifest = _json_object(self.model_dir / "sha256.json")
                if manifest.get(CHECKPOINT_FILENAME) != checkpoint_hash:
                    raise GroundingSpecialistError("Grounding specialist checksum manifest mismatch")
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or not isinstance(payload.get("specialist_state_dict"), dict):
                    raise GroundingSpecialistError("Grounding specialist checkpoint lacks specialist_state_dict")
                if payload.get("model_name") != MODEL_NAME:
                    raise GroundingSpecialistError("Grounding specialist model name mismatch")
                if payload.get("base_model") != BASE_MODEL:
                    raise GroundingSpecialistError("Grounding specialist base model mismatch")
                if payload.get("training_recipe_version") != TRAINING_RECIPE_VERSION:
                    raise GroundingSpecialistError("Grounding specialist training recipe mismatch")
                if int(payload.get("step", -1)) != CHECKPOINT_STEP:
                    raise GroundingSpecialistError("Grounding specialist checkpoint step mismatch")
                head = _build_head(torch)
                head.load_state_dict(payload["specialist_state_dict"], strict=True)
                head.to(device).eval()
                self._head = head
                self._device = device
                self._state = "ready"
                self._load_count += 1
                self.checkpoint_hash = checkpoint_hash
                logger.info(
                    "Loaded Grounding Specialist v1.1 checkpoint=%s step=%d device=%s",
                    checkpoint_hash,
                    CHECKPOINT_STEP,
                    device,
                )
            except Exception as error:
                self._head = None
                self._device = None
                self._state = "failed"
                message = str(error).lower()
                if isinstance(error, FileNotFoundError):
                    self._error = f"Grounding Specialist v1.1 model file missing: {Path(error.filename).name if error.filename else CHECKPOINT_FILENAME}."
                elif "hash mismatch" in message or "checksum" in message:
                    self._error = "Grounding Specialist v1.1 checkpoint checksum mismatch."
                elif "state_dict" in message or "size mismatch" in message or "model name mismatch" in message or "base model mismatch" in message:
                    self._error = "Grounding Specialist v1.1 checkpoint is incompatible with the expected architecture."
                else:
                    self._error = "Grounding Specialist v1.1 initialization failed."
                logger.warning(
                    "Grounding Specialist v1.1 could not load; using original Grounding DINO scores: %s",
                    error,
                )
                if isinstance(error, GroundingSpecialistError):
                    raise
                raise GroundingSpecialistError("Grounding Specialist v1.1 could not load") from error

    def score_proposals(self, hidden_state: Any, base_boxes_cxcywh: Any) -> Any:
        device = str(hidden_state.device)
        self.load(device)
        if self._head is None:
            raise GroundingSpecialistError("Grounding Specialist v1.1 is unavailable")
        import torch

        with torch.inference_mode():
            query_logits, _unused_refined_boxes = self._head(hidden_state, base_boxes_cxcywh)
            return query_logits.sigmoid()

    def reset_for_tests(self) -> None:
        with self._load_lock:
            self._head = None
            self._device = None
            self._state = "unloaded" if self._enabled else "disabled"
            self._error = None
            self._load_count = 0
            self.checkpoint_hash = None


_GROUNDING_SPECIALIST = GroundingSpecialist()


def get_grounding_specialist() -> GroundingSpecialist:
    return _GROUNDING_SPECIALIST
