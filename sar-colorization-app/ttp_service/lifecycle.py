"""Thread-safe, once-per-process TTP model lifecycle."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .inference import TTPInferenceEngine, TTPInferenceError
from .schemas import CHECKPOINT_NAME, CHECKPOINT_SHA256


class LifecycleError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _default_engine() -> TTPInferenceEngine:
    repository = Path(os.getenv("TTP_REPOSITORY_DIR", "/opt/ttp/TTP"))
    checkpoint = Path(os.getenv("TTP_CHECKPOINT_PATH", "/opt/ttp/assets/epoch_260.pth"))
    return TTPInferenceEngine(repository, checkpoint)


class ModelLifecycle:
    def __init__(self, engine_factory: Callable[[], TTPInferenceEngine] = _default_engine) -> None:
        self._factory = engine_factory
        self._engine: TTPInferenceEngine | None = None
        self._state = "unloaded"
        self._failure_code: str | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.model_load_ms = 0
        self.model_load_count = 0
        self.inference_count = 0
        self.model_reuse_count = 0
        self._verification_result: bool | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def checkpoint_verified(self) -> bool:
        if self._verification_result is not None:
            return self._verification_result
        from .verify_assets import verify_checkpoint
        try:
            verify_checkpoint(Path(os.getenv("TTP_CHECKPOINT_PATH", "/opt/ttp/assets/epoch_260.pth")), CHECKPOINT_SHA256)
            self._verification_result = True
        except Exception:
            self._verification_result = False
        return self._verification_result

    def checkpoint(self) -> str:
        return CHECKPOINT_NAME

    def device(self) -> str | None:
        return "cuda" if self._state == "ready" else None

    def health(self) -> dict[str, object]:
        return {
            "status": "ready" if self._state == "ready" else "failed" if self._state == "failed" else self._state,
            "lifecycle": self._state,
            "device": self.device(),
            "checkpoint": self.checkpoint(),
            "loaded": self._state == "ready",
            "load_count": self.model_load_count,
            "reuse_count": self.model_reuse_count,
            "inference_count": self.inference_count,
            "errors": [self._failure_code] if self._failure_code else [],
        }

    @property
    def probability_source(self) -> str | None:
        value = getattr(self._engine, "last_probability_source", None) if self._engine is not None else None
        return str(value) if value else None

    def load(self) -> None:
        if self._state == "ready":
            return
        if self._state == "failed":
            raise LifecycleError("MODEL_LOAD_FAILED", "TTP model loading previously failed; restart after correcting verified assets or CUDA.")
        with self._load_lock:
            if self._state == "ready":
                return
            if self._state == "failed":
                raise LifecycleError("MODEL_LOAD_FAILED", "TTP model loading previously failed; restart after correcting verified assets or CUDA.")
            self._state = "loading"
            started = time.perf_counter()
            try:
                if not self.checkpoint_verified:
                    raise TTPInferenceError("Checkpoint verification failed.")
                engine = self._factory()
                engine.load()
                self._engine = engine
                self.model_load_ms = max(0, round((time.perf_counter() - started) * 1000))
                self.model_load_count += 1
                self._failure_code = None
                self._state = "ready"
            except Exception as error:
                self._engine = None
                self._state = "failed"
                self._failure_code = "MODEL_LOAD_FAILED"
                raise LifecycleError("MODEL_LOAD_FAILED", "TTP model loading failed; verified assets and CUDA are required.") from error

    def predict(self, earlier: bytes, later: bytes, earlier_suffix: str, later_suffix: str) -> tuple[np.ndarray, bool, int, dict[str, float | None]]:
        if self._state == "unloaded":
            self.load()
        if self._state != "ready" or self._engine is None:
            raise LifecycleError("MODEL_NOT_READY", "TTP is not ready for inference.")
        with self._inference_lock:
            reused = self.inference_count > 0
            started = time.perf_counter()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                mask = self._engine.predict(earlier, later, earlier_suffix, later_suffix)
                inference_ms = max(0, round((time.perf_counter() - started) * 1000))
                self.inference_count += 1
                if reused:
                    self.model_reuse_count += 1
                metrics = {
                    "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 2) if torch.cuda.is_available() else None,
                    "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 2) if torch.cuda.is_available() else None,
                    "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1048576, 2) if torch.cuda.is_available() else None,
                }
                return mask, reused, inference_ms, metrics
            except Exception as error:
                try:
                    import torch
                    is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    is_oom = "out of memory" in str(error).lower()
                code = "CUDA_OOM" if is_oom else "INFERENCE_FAILED"
                raise LifecycleError(code, "TTP inference could not complete safely.") from error


MODEL_LIFECYCLE = ModelLifecycle()
