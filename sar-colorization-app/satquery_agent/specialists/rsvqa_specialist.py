"""RSVQA Specialist v1 inference over the shared frozen SVE OpenCLIP encoder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from PIL import Image

from ..models import QuestionCategory, SpecialistHealth
from ..services.sve_service import SVEManager, get_sve_service
from ..sve_artifacts import EXPECTED_EMBEDDING_DIMENSION, SVEError
from .vqa import get_vqa


LOGGER = logging.getLogger("satquery.rsvqa_specialist")
MODEL_NAME = "SatQuery RSVQA Specialist v1"
MODEL_USED = "RSVQA Specialist v1"
CHECKPOINT_FILENAME = "rsvqa_specialist_v1_head.pt"
VOCAB_FILENAME = "task_answer_vocab.json"
EXPECTED_CHECKPOINT_SHA256 = "71c0ab56ee650813bd495e8a3bc777353b6907a097af860e417f60523efe56ad"
TASKS = ("presence", "comp", "rural_urban", "count")
TASK_CLASS_COUNTS = {"presence": 2, "comp": 2, "rural_urban": 2, "count": 202}
TASK_BY_CATEGORY = {
    QuestionCategory.PRESENCE_VQA: "presence",
    QuestionCategory.COMPARISON_VQA: "comp",
    QuestionCategory.RURAL_URBAN_CLASSIFICATION: "rural_urban",
    QuestionCategory.COUNT_VQA: "count",
}


class RSVQASpecialistError(RuntimeError):
    """A safe integration failure that permits deterministic VQA fallback."""


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def configured_model_dir() -> Path:
    configured = os.getenv("RSVQA_SPECIALIST_MODEL_DIR", "models/rsvqa_specialist_v1").strip()
    if not configured:
        raise RSVQASpecialistError("RSVQA_SPECIALIST_MODEL_DIR is empty")
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = Path(__file__).resolve().parents[2] / candidate
    return candidate.resolve()


def rsvqa_specialist_configured() -> bool:
    if not _bool_env("RSVQA_SPECIALIST_ENABLED", True):
        return False
    try:
        directory = configured_model_dir()
    except RSVQASpecialistError:
        return False
    return (
        all(importlib.util.find_spec(name) is not None for name in ("torch", "open_clip", "PIL"))
        and all((directory / name).is_file() for name in (
            CHECKPOINT_FILENAME,
            VOCAB_FILENAME,
            "preprocessing.json",
            "integration_contract.json",
            "validation_metrics.json",
        ))
    )


def classify_rsvqa_task(question: str) -> Optional[str]:
    """Return the exported task-head name for a supported RSVQA-style question."""
    return TASK_BY_CATEGORY.get(get_vqa().classify_question(question).category)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RSVQASpecialistError(f"Could not read {path.name}") from error
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RSVQASpecialistError(f"Invalid RSVQA artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise RSVQASpecialistError(f"RSVQA artifact must contain an object: {path.name}")
    return value


def _build_head(architecture: Mapping[str, Any]) -> Any:
    """Construct the exact compact head used for specialist training."""
    import torch

    image_dim = int(architecture.get("image_dim", -1))
    text_dim = int(architecture.get("text_dim", -1))
    hidden_dim = int(architecture.get("hidden_dim", -1))
    dropout = float(architecture.get("dropout", -1))
    if (image_dim, text_dim, hidden_dim, dropout) != (768, 768, 768, 0.15):
        raise RSVQASpecialistError("Checkpoint architecture does not match RSVQA Specialist v1")
    if int(architecture.get("count_classes", -1)) != 202:
        raise RSVQASpecialistError("Checkpoint count-head shape is unsupported")

    class RSVQAFusionHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_projection = torch.nn.Sequential(
                torch.nn.Linear(image_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
            )
            self.text_projection = torch.nn.Sequential(
                torch.nn.Linear(text_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
            )
            self.fusion = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 4, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            self.presence_head = torch.nn.Linear(hidden_dim, 2)
            self.comparison_head = torch.nn.Linear(hidden_dim, 2)
            self.rural_urban_head = torch.nn.Linear(hidden_dim, 2)
            self.count_head = torch.nn.Linear(hidden_dim, 202)

        def forward(self, image_features: Any, text_features: Any, task: str) -> Any:
            image_projected = self.image_projection(image_features)
            text_projected = self.text_projection(text_features)
            shared = self.fusion(torch.cat((
                image_projected,
                text_projected,
                image_projected * text_projected,
                torch.abs(image_projected - text_projected),
            ), dim=-1))
            head = {
                "presence": self.presence_head,
                "comp": self.comparison_head,
                "rural_urban": self.rural_urban_head,
                "count": self.count_head,
            }.get(task)
            if head is None:
                raise RSVQASpecialistError(f"Unsupported RSVQA task: {task}")
            return head(shared)

    return RSVQAFusionHead()


class RSVQASpecialist:
    def __init__(
        self,
        *,
        sve_manager: Optional[SVEManager] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.sve_manager = sve_manager or get_sve_service()
        self.model_dir = (model_dir or configured_model_dir()).resolve()
        self._load_lock = threading.RLock()
        self._head: Any = None
        self._vocabulary: dict[str, list[str]] = {}
        self._device: Optional[str] = None
        self._state = "unloaded" if _bool_env("RSVQA_SPECIALIST_ENABLED", True) else "disabled"
        self._error: Optional[str] = None
        self.checkpoint_hash: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._state == "ready" and self._head is not None

    def health(self) -> SpecialistHealth:
        return SpecialistHealth(status=self._state, device=self._device, error=self._error)

    def load(self) -> None:
        if self.loaded:
            return
        if not _bool_env("RSVQA_SPECIALIST_ENABLED", True):
            self._state = "disabled"
            raise RSVQASpecialistError("RSVQA Specialist is disabled")
        with self._load_lock:
            if self.loaded:
                return
            try:
                import torch

                checkpoint_path = self.model_dir / CHECKPOINT_FILENAME
                vocabulary_path = self.model_dir / VOCAB_FILENAME
                actual_hash = _sha256(checkpoint_path)
                if actual_hash != EXPECTED_CHECKPOINT_SHA256:
                    raise RSVQASpecialistError("RSVQA specialist checkpoint hash mismatch")
                manifest = _json_object(self.model_dir / "sha256.json")
                if manifest.get(CHECKPOINT_FILENAME) != actual_hash:
                    raise RSVQASpecialistError("RSVQA checksum manifest does not match the checkpoint")
                if manifest.get(VOCAB_FILENAME) != _sha256(vocabulary_path):
                    raise RSVQASpecialistError("RSVQA vocabulary hash mismatch")
                for artifact_name in ("preprocessing.json", "integration_contract.json", "validation_metrics.json"):
                    if manifest.get(artifact_name) != _sha256(self.model_dir / artifact_name):
                        raise RSVQASpecialistError(f"RSVQA artifact hash mismatch: {artifact_name}")
                vocabulary_payload = _json_object(vocabulary_path)
                labels = vocabulary_payload.get("task_labels")
                if not isinstance(labels, dict) or set(labels) != set(TASKS):
                    raise RSVQASpecialistError("RSVQA task vocabulary is malformed")
                vocabulary = {task: list(labels[task]) for task in TASKS}
                if any(len(vocabulary[task]) != TASK_CLASS_COUNTS[task] for task in TASKS):
                    raise RSVQASpecialistError("RSVQA task vocabulary sizes do not match the exported heads")
                preprocessing = _json_object(self.model_dir / "preprocessing.json")
                image_preprocessing = preprocessing.get("image_preprocessing") or {}
                question_preprocessing = preprocessing.get("question_preprocessing") or {}
                if image_preprocessing.get("input_size") != [3, 224, 224] or image_preprocessing.get("openclip_transform") != "OpenCLIP ViT-L-14 validation transform":
                    raise RSVQASpecialistError("RSVQA image preprocessing contract is invalid")
                if question_preprocessing.get("tokenizer") != "OpenCLIP ViT-L-14 tokenizer" or question_preprocessing.get("context_length") != 77:
                    raise RSVQASpecialistError("RSVQA question preprocessing contract is invalid")
                integration = _json_object(self.model_dir / "integration_contract.json")
                if integration.get("display_name") != MODEL_NAME or integration.get("supported_question_types") != list(TASKS):
                    raise RSVQASpecialistError("RSVQA integration contract is invalid")
                if (integration.get("input") or {}).get("task_id_mapping") != {"presence": 0, "comp": 1, "rural_urban": 2, "count": 3}:
                    raise RSVQASpecialistError("RSVQA task mapping is invalid")

                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or payload.get("model_name") != MODEL_NAME:
                    raise RSVQASpecialistError("RSVQA specialist checkpoint metadata is invalid")
                architecture = payload.get("architecture")
                state = payload.get("specialist_state_dict")
                if not isinstance(architecture, dict) or not isinstance(state, dict):
                    raise RSVQASpecialistError("RSVQA specialist checkpoint is incomplete")
                if architecture.get("backbone") != "OpenCLIP ViT-L-14" or architecture.get("vision_adapter") != "SatQuery Vision Encoder v1":
                    raise RSVQASpecialistError("RSVQA specialist backbone contract is invalid")
                if architecture.get("backbone_pretrained") != "laion2b_s32b_b82k" or int(architecture.get("question_context_length", -1)) != 77:
                    raise RSVQASpecialistError("RSVQA specialist preprocessing contract is invalid")
                if payload.get("task_labels") != labels:
                    raise RSVQASpecialistError("Checkpoint labels differ from the exported vocabulary")

                self.sve_manager.load()
                sve_health = self.sve_manager.health()
                if sve_health.status != "ready" or not sve_health.device:
                    raise RSVQASpecialistError("Frozen SatQuery Vision Encoder is unavailable")
                head = _build_head(architecture)
                head.load_state_dict(state, strict=True)
                head.eval()
                for parameter in head.parameters():
                    parameter.requires_grad_(False)
                head.to(sve_health.device, dtype=torch.float32)

                self._head = head
                self._vocabulary = vocabulary
                self._device = sve_health.device
                self.checkpoint_hash = actual_hash
                self._state = "ready"
                self._error = None
                LOGGER.info("Loaded RSVQA Specialist")
                LOGGER.info("Checkpoint hash: sha256:%s", actual_hash)
                LOGGER.info("Vocabulary size: %d", sum(len(values) for values in vocabulary.values()))
                LOGGER.info("Task heads loaded: %s", ", ".join(TASKS))
                LOGGER.info("Device: %s", self._device)
            except (RSVQASpecialistError, SVEError) as error:
                self._head = None
                self._state = "failed"
                self._error = str(error)
                raise RSVQASpecialistError(str(error)) from error
            except Exception as error:
                self._head = None
                self._state = "failed"
                self._error = "RSVQA Specialist v1 could not be loaded"
                raise RSVQASpecialistError(self._error) from error

    def predict(
        self,
        image: Image.Image,
        question: str,
        *,
        content_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        task = classify_rsvqa_task(question)
        if task is None:
            raise RSVQASpecialistError("Question is outside the exported RSVQA task families")
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise RSVQASpecialistError("A valid image is required")
        started = time.perf_counter()
        self.load()
        try:
            import torch

            image_features, text_features, encoder = self.sve_manager.encode_shared_features(
                image.convert("RGB"), question, content_hash=content_hash
            )
            target_device = str(encoder.get("device") or self._device or "cpu")
            if target_device != self._device:
                self._head.to(target_device, dtype=torch.float32)
                self._device = target_device
            with torch.inference_mode():
                logits_tensor = self._head(
                    image_features.to(self._device, dtype=torch.float32),
                    text_features.to(self._device, dtype=torch.float32),
                    task,
                )[0].float().cpu()
                probabilities_tensor = torch.softmax(logits_tensor, dim=-1)
            answer_id = int(torch.argmax(probabilities_tensor).item())
            logits = [float(value) for value in logits_tensor.tolist()]
            probabilities = [float(value) for value in probabilities_tensor.tolist()]
            return {
                "answer": self._vocabulary[task][answer_id],
                "confidence": probabilities[answer_id],
                "task": task,
                "logits": logits,
                "probabilities": probabilities,
                "model_used": MODEL_USED,
                "runtime_ms": max(0, round((time.perf_counter() - started) * 1000)),
                "encoder": encoder,
            }
        except RSVQASpecialistError:
            raise
        except Exception as error:
            raise RSVQASpecialistError("RSVQA Specialist v1 prediction failed") from error


RSVQA_SPECIALIST = RSVQASpecialist()


def get_rsvqa_specialist() -> RSVQASpecialist:
    return RSVQA_SPECIALIST
