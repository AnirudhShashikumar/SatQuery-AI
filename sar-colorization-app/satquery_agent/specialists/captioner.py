"""Lazy local remote-sensing image captioning specialist."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..models import (
    CaptionResult,
    Confidence,
    ConfidenceLevel,
    ImageMetadata,
    Modality,
    ModelProvenance,
    SpecialistHealth,
)


CHECKPOINT = "Gurveer05/blip-image-captioning-base-rscid-finetuned"
BASE_ARCHITECTURE = "Salesforce BLIP image-captioning base"
ADAPTATION_DATASET = "RSICD (Remote Sensing Image Caption Dataset)"
MODEL_LICENSE = "Apache-2.0"
MODEL_SOURCE = "https://huggingface.co/Gurveer05/blip-image-captioning-base-rscid-finetuned"
SUPPORTED_MODALITIES = {Modality.OPTICAL, Modality.MULTISPECTRAL}
MAX_NEW_TOKENS = 40
NUM_BEAMS = 3
LIMITATIONS = [
    "The caption is model-generated and may omit or misidentify small objects.",
    "The checkpoint was adapted on 224×224 optical aerial imagery from RSICD.",
    "The result is not a substitute for expert remote-sensing validation.",
]


class CaptionerError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def captioner_enabled() -> bool:
    return os.getenv("SATQUERY_CAPTIONER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def captioner_dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors"))


def captioner_configured() -> bool:
    return captioner_enabled() and bool(os.getenv("SATQUERY_CAPTIONER_CHECKPOINT", CHECKPOINT).strip()) and captioner_dependencies_available()


class RemoteSensingCaptioner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "unloaded"
        self._device: Optional[str] = None
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._safe_error: Optional[str] = None
        self._load_count = 0

    @property
    def checkpoint(self) -> str:
        return os.getenv("SATQUERY_CAPTIONER_CHECKPOINT", CHECKPOINT).strip()

    @property
    def cache_dir(self) -> str:
        configured = os.getenv("SATQUERY_MODEL_CACHE", "").strip()
        return configured or str(Path(tempfile.gettempdir()) / "geovision-satquery-hf")

    @property
    def limitations(self) -> list[str]:
        return list(LIMITATIONS)

    def is_available(self) -> bool:
        return captioner_configured() and self._state != "failed"

    def health(self) -> SpecialistHealth:
        if not captioner_configured():
            return SpecialistHealth(status="disabled", device=None, error=None)
        return SpecialistHealth(status=self._state, device=self._device, error=self._safe_error)

    def _select_device(self, torch: Any) -> str:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> None:
        if self._state == "ready":
            return
        if not captioner_configured():
            raise CaptionerError("CAPTIONER_UNAVAILABLE", "The remote-sensing captioner is not configured in this local environment.")
        with self._lock:
            if self._state == "ready":
                return
            if self._state == "failed":
                raise CaptionerError("CAPTIONER_LOAD_FAILED", "The local captioning checkpoint could not be loaded.")
            self._state = "loading"
            self._safe_error = None
            try:
                import torch
                from transformers import BlipForConditionalGeneration, BlipProcessor

                device = self._select_device(torch)
                processor = BlipProcessor.from_pretrained(self.checkpoint, cache_dir=self.cache_dir)
                model = BlipForConditionalGeneration.from_pretrained(self.checkpoint, cache_dir=self.cache_dir)
                model.eval()
                model.to(device)
                self._processor = processor
                self._model = model
                self._device = device
                self._load_count += 1
                self._state = "ready"
            except Exception as error:
                self._processor = None
                self._model = None
                self._device = None
                self._state = "failed"
                self._safe_error = "Checkpoint loading failed."
                raise CaptionerError("CAPTIONER_LOAD_FAILED", "The local captioning checkpoint could not be loaded.") from error

    def describe(
        self,
        image: Image.Image,
        metadata: ImageMetadata,
        modality: Modality,
        bands_used: list[str],
        image_representation: str,
    ) -> CaptionResult:
        if modality not in SUPPORTED_MODALITIES:
            raise CaptionerError(
                "UNSUPPORTED_CAPTION_MODALITY",
                "The connected RSICD captioner supports optical and multispectral RGB representations only; SAR captioning is not implemented.",
            )
        if metadata.band_count < 3:
            raise CaptionerError(
                "UNSUPPORTED_CAPTION_BANDS",
                "The connected captioner requires an RGB or RGB-like optical representation with at least three bands.",
            )
        if metadata.band_count > 3 and len(metadata.selected_visual_bands) < 3:
            raise CaptionerError(
                "UNSUPPORTED_CAPTION_BANDS",
                "Captioning requires an explicit RGB band mapping for multispectral rasters; no mapping was guessed.",
            )

        was_ready = self._state == "ready"
        load_started = time.perf_counter()
        self.load()
        model_load_ms = 0 if was_ready else max(0, round((time.perf_counter() - load_started) * 1000))
        if self._processor is None or self._model is None or self._device is None:
            raise CaptionerError("CAPTIONER_UNAVAILABLE", "The local captioning specialist is unavailable.")

        import torch

        inference_started = time.perf_counter()
        try:
            inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {name: value.to(self._device) for name, value in inputs.items()}
            with torch.inference_mode():
                tokens = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=NUM_BEAMS,
                    do_sample=False,
                )
            caption = self._processor.decode(tokens[0], skip_special_tokens=True).strip()
        except Exception as error:
            raise CaptionerError("CAPTION_INFERENCE_FAILED", "The local captioning model could not generate a description safely.") from error
        runtime_ms = max(0, round((time.perf_counter() - inference_started) * 1000))
        if not caption:
            raise CaptionerError("EMPTY_CAPTION", "The captioning model returned an empty description.")

        warnings = list(metadata.warnings)
        confidence_level = ConfidenceLevel.MODERATE
        if modality == Modality.MULTISPECTRAL and bands_used == ["band_1", "band_2", "band_3"]:
            confidence_level = ConfidenceLevel.LOW
            warnings.append("Multispectral band roles were unavailable; the model used the first three display bands as an RGB-like representation.")
        confidence = Confidence(
            level=confidence_level,
            score=None,
            reason=(
                "The captioning model does not provide calibrated probability estimates. "
                "Confidence reflects supported optical input and band-mapping certainty only."
            ),
        )
        return CaptionResult(
            caption=caption,
            confidence=confidence,
            model=ModelProvenance(
                tool_id="rs_captioner",
                checkpoint=self.checkpoint,
                base_architecture=BASE_ARCHITECTURE,
                adaptation_dataset=ADAPTATION_DATASET,
                remote_sensing_adapted=True,
                license=MODEL_LICENSE,
                source=MODEL_SOURCE,
            ),
            warnings=list(dict.fromkeys(warnings)),
            runtime_ms=runtime_ms,
            device=self._device,
            image_representation=image_representation,
            bands_used=bands_used,
            model_load_ms=model_load_ms,
            reused_model=was_ready,
        )

    def reset_for_tests(self) -> None:
        with self._lock:
            self._state = "unloaded"
            self._device = None
            self._processor = None
            self._model = None
            self._safe_error = None
            self._load_count = 0


_CAPTIONER = RemoteSensingCaptioner()


def get_captioner() -> RemoteSensingCaptioner:
    return _CAPTIONER
