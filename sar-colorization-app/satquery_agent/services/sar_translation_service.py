"""Shared, optional SAR-to-optical translation lifecycle for agent evidence."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from ..image_ingestion import ImageIngestionError, save_preview
from ..models import ImageMetadata, ImageModality, RepresentationType, SpecialistHealth


LOGGER = logging.getLogger("satquery.sar_translation")
DISCLOSURE = (
    "This optical-like image is generated from SAR by a learned translation model. "
    "It is not an observed optical image and may contain hallucinated, omitted, or spatially distorted features."
)
PIX2PIX_DISCLOSURE = (
    "Learned optical-like representation generated from SAR by Pix2Pix; not observed optical imagery and not ground truth. "
    "It is not an observed optical image and may contain hallucinated, omitted, or spatially distorted features."
)
GROUNDING_DISCLOSURE = (
    "Bounding boxes were inferred on the generated optical-like representation. They are not direct detections "
    "from the original SAR measurement."
)
RSVQA_DISCLOSURE = (
    "The answer was inferred from the generated optical-like representation and should be treated as "
    "supporting evidence, not ground truth."
)
ALLOWED_MODELS = {"sarfusionformer", "pix2pix"}


class SarTranslationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _checkpoint_path(name: str, default: Path) -> Path:
    configured = os.getenv(name)
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2].parent / candidate
        return candidate.resolve()
    return default.resolve()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SARFUSIONFORMER_CHECKPOINT = _checkpoint_path(
    "SARFUSIONFORMER_CHECKPOINT", WORKSPACE_ROOT / "models/checkpoints/sarfusionformer_256_decoder_best.pt"
)
PIX2PIX_CHECKPOINT = _checkpoint_path("PIX2PIX_CHECKPOINT", WORKSPACE_ROOT / "pix2pix_gen_180.pth")
COLOR_CORRECTOR_CHECKPOINT = _checkpoint_path(
    "COLOR_CORRECTOR_CHECKPOINT", WORKSPACE_ROOT / "models/checkpoints/color_corrector_256_best.pt"
)


def translation_enabled() -> bool:
    return _bool_env("SATQUERY_SAR_TRANSLATION_ENABLED", False)


def optical_specialists_enabled() -> bool:
    return _bool_env("SATQUERY_SAR_TRANSLATION_OPTICAL_SPECIALISTS_ENABLED", True)


def save_artifacts_enabled() -> bool:
    return _bool_env("SATQUERY_SAR_TRANSLATION_SAVE_ARTIFACTS", True)


def selected_translation_model() -> str:
    return os.getenv("SATQUERY_SAR_TRANSLATION_MODEL", "sarfusionformer").strip().lower()


@dataclass(frozen=True)
class TranslationEligibility:
    eligible: bool
    requested_model: str
    selected_model: Optional[str]
    fallback_model: Optional[str]
    fallback_eligible: bool
    reason: Optional[str]
    finite_value_ratio: float
    width: int
    height: int
    band_count: int
    dtype: str
    channel_interpretation: str
    preprocessing_method: str


@dataclass
class SarTranslationResult:
    image: Image.Image
    width: int
    height: int
    model_used: str
    fallback_used: bool
    fallback_reason: Optional[str]
    color_correction_used: bool
    device: str
    runtime_ms: int
    stage_durations_ms: dict[str, int]
    preprocessing_method: str
    input_channel_interpretation: str
    output_value_range: list[float]
    warnings: list[str]
    provenance: dict[str, Any]
    artifact_url: Optional[str]
    normalized_sar_preview_url: Optional[str]
    color_corrected_artifact_url: Optional[str]
    content_hash: str
    stage_statuses: dict[str, str] = field(default_factory=dict)


def _elapsed(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _finite_ratio(raster: np.ndarray, metadata: ImageMetadata) -> float:
    values = np.asarray(raster)
    if values.size == 0 or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        return 0.0
    finite = np.isfinite(values)
    if metadata.nodata is not None and math.isfinite(metadata.nodata):
        finite &= ~np.isclose(values, metadata.nodata, equal_nan=False)
    return float(finite.mean())


def _sarfusionformer_eligibility(raster: np.ndarray, metadata: ImageMetadata) -> tuple[bool, str]:
    values = np.asarray(raster)
    if metadata.representation != RepresentationType.SCIENTIFIC_RASTER:
        return False, "SARFusionFormer requires a scientific raster with a verified VV/VH pair."
    if metadata.effective_modality != ImageModality.SAR_VV_VH:
        return False, "SARFusionFormer requires the effective modality to be a verified VV/VH pair."
    if metadata.band_count != 2 or values.ndim != 3 or values.shape[2] != 2:
        return False, "SARFusionFormer requires exactly two source SAR channels."
    descriptions = " ".join(value or "" for value in metadata.band_descriptions).lower()
    explicitly_confirmed = metadata.user_confirmed_modality == ImageModality.SAR_VV_VH
    described_pair = "vv" in descriptions and "vh" in descriptions
    if not (explicitly_confirmed or described_pair):
        return False, "SARFusionFormer requires a verified VV/VH channel order; the two bands are ambiguous."
    if _finite_ratio(values, metadata) <= 0:
        return False, "The SAR input contains no finite, non-NoData values."
    return True, "Verified channel mapping: channel 1=VV, channel 2=VH."


def _pix2pix_eligibility(raster: np.ndarray, metadata: ImageMetadata, model_image: Image.Image) -> tuple[bool, str]:
    if metadata.effective_modality not in {
        ImageModality.SAR_PREVIEW, ImageModality.SAR_VV, ImageModality.SAR_VH, ImageModality.SAR_VV_VH
    }:
        return False, "Pix2Pix translation requires an explicitly identified SAR representation."
    if metadata.band_count not in {1, 2, 3, 4}:
        return False, "Pix2Pix supports the existing one- to four-band SAR display rendering only."
    if model_image.width <= 0 or model_image.height <= 0:
        return False, "A valid SAR display image is unavailable."
    if _finite_ratio(np.asarray(raster), metadata) <= 0:
        return False, "The SAR input contains no finite, non-NoData values."
    return True, (
        "Existing ingestion display converted to RGB; a one-band display is explicitly replicated to RGB "
        "and a two-band input uses the documented ingestion false-color rendering."
    )


def evaluate_translation_eligibility(
    raster: np.ndarray,
    metadata: ImageMetadata,
    model_image: Image.Image,
    requested_model: Optional[str] = None,
) -> TranslationEligibility:
    requested = (requested_model or selected_translation_model()).strip().lower()
    values = np.asarray(raster)
    height = int(values.shape[0]) if values.ndim >= 2 else 0
    width = int(values.shape[1]) if values.ndim >= 2 else 0
    ratio = _finite_ratio(values, metadata)
    if requested not in ALLOWED_MODELS:
        return TranslationEligibility(False, requested, None, None, False, f"Unsupported SAR translation model: {requested}.", ratio, width, height, metadata.band_count, metadata.dtype, "unavailable", "unavailable")
    checks = {
        "sarfusionformer": _sarfusionformer_eligibility(raster, metadata),
        "pix2pix": _pix2pix_eligibility(raster, metadata, model_image),
    }
    fallback = "pix2pix" if requested == "sarfusionformer" else "sarfusionformer"
    eligible, reason = checks[requested]
    fallback_eligible, fallback_reason = checks[fallback]
    selected = requested if eligible else fallback if fallback_eligible else None
    channel_interpretation = reason if eligible else fallback_reason if fallback_eligible else reason
    preprocessing = (
        "finite per-channel 1st/99th percentile normalization; VV/VH resize to 256x256"
        if selected == "sarfusionformer"
        else "existing ingestion SAR display converted to RGB; bicubic resize to 256x256; scale to [-1,1]"
        if selected == "pix2pix" else "unavailable"
    )
    no_model_reason = None if selected else (
        "SAR-to-optical translation was skipped because neither configured translator accepts this input. "
        f"{reason} {fallback_reason}"
    )
    return TranslationEligibility(
        eligible=selected is not None,
        requested_model=requested,
        selected_model=selected,
        fallback_model=fallback,
        fallback_eligible=fallback_eligible,
        reason=no_model_reason if not selected else (None if selected == requested else reason),
        finite_value_ratio=ratio,
        width=width,
        height=height,
        band_count=metadata.band_count,
        dtype=metadata.dtype,
        channel_interpretation=channel_interpretation,
        preprocessing_method=preprocessing,
    )


class SarTranslationService:
    def __init__(
        self,
        *,
        model_factories: Optional[dict[str, Callable[[], Any]]] = None,
        color_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._load_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._models: dict[str, Any] = {}
        self._model_errors: dict[str, str] = {}
        self._color_corrector: Any = None
        self._color_error: Optional[str] = None
        self._state = "disabled" if not translation_enabled() else "unloaded"
        self._device: Optional[str] = None
        self._safe_error: Optional[str] = None
        self._load_count = 0
        self._reuse_count = 0
        self._automatic_preview_active = False
        self._model_factories = model_factories or {}
        self._color_factory = color_factory

    @property
    def enabled(self) -> bool:
        return translation_enabled() or self._automatic_preview_active

    @property
    def state(self) -> str:
        if not self.enabled:
            return "disabled"
        return "unloaded" if self._state == "disabled" else self._state

    def _select_device(self) -> str:
        import torch

        requested = os.getenv("SATQUERY_SAR_TRANSLATION_DEVICE", "auto").strip().lower()
        if requested not in {"auto", "cuda", "mps", "cpu"}:
            raise SarTranslationError("INVALID_DEVICE", "The configured SAR translation device is unsupported.")
        cuda = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(mps_backend is not None and mps_backend.is_available())
        if requested == "cuda" and not cuda:
            raise SarTranslationError("DEVICE_UNAVAILABLE", "The configured CUDA device is unavailable.")
        if requested == "mps" and not mps:
            raise SarTranslationError("DEVICE_UNAVAILABLE", "The configured MPS device is unavailable.")
        return requested if requested != "auto" else "cuda" if cuda else "mps" if mps else "cpu"

    def _color_configured(self) -> bool:
        return self._color_corrector is not None or self._color_factory is not None or COLOR_CORRECTOR_CHECKPOINT.is_file()

    def use_color_correction(self) -> bool:
        return _bool_env("SATQUERY_SAR_TRANSLATION_USE_COLOR_CORRECTION", self._color_configured())

    def health(self) -> SpecialistHealth:
        return SpecialistHealth(
            status=self.state,
            device=self._device,
            error=self._safe_error,
        )

    def health_payload(self) -> dict[str, Any]:
        return {
            **self.health().model_dump(mode="json"),
            "enabled": self.enabled,
            "selected_model": "pix2pix" if self._automatic_preview_active and not translation_enabled() else selected_translation_model(),
            "fallback_available": "pix2pix" in self._models or PIX2PIX_CHECKPOINT.is_file(),
            "color_corrector_available": self._color_configured(),
            "load_count": self._load_count,
            "reuse_count": self._reuse_count,
        }

    def register_external_models(
        self,
        *,
        pix2pix: Any = None,
        sarfusionformer: Any = None,
        color_corrector: Any = None,
        pix2pix_error: Optional[str] = None,
        sarfusionformer_error: Optional[str] = None,
        color_error: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        with self._load_lock:
            supplied = {"pix2pix": pix2pix, "sarfusionformer": sarfusionformer}
            for name, model in supplied.items():
                if model is not None:
                    existing = self._models.get(name)
                    if existing is not None and existing is not model:
                        LOGGER.warning("Ignored a second registered %s model instance", name)
                        continue
                    self._models[name] = model
            self._model_errors.update({
                name: error for name, error in {
                    "pix2pix": pix2pix_error, "sarfusionformer": sarfusionformer_error
                }.items() if error
            })
            if color_corrector is not None:
                if self._color_corrector is None or self._color_corrector is color_corrector:
                    self._color_corrector = color_corrector
                else:
                    LOGGER.warning("Ignored a second registered SAR color-corrector instance")
            self._color_error = color_error
            if device:
                self._device = str(device)
            if self._models and self.enabled:
                self._state = "ready"

    def _load_model(self, name: str) -> Any:
        import torch

        if name in self._models:
            self._reuse_count += 1
            return self._models[name]
        if name in self._model_errors and name not in self._model_factories:
            raise SarTranslationError("MODEL_UNAVAILABLE", f"The host could not load the {name} translation model.")
        factory = self._model_factories.get(name)
        device = self._device or self._select_device()
        self._device = device
        if factory is not None:
            model = factory()
        elif name == "sarfusionformer":
            if not SARFUSIONFORMER_CHECKPOINT.is_file():
                raise SarTranslationError("CHECKPOINT_MISSING", "The SARFusionFormer checkpoint is unavailable.")
            from sarfusionformer import SARFusionFormer

            model = SARFusionFormer(input_channels=2, output_channels=3, base_channels=48, transformer_depth=4, attention_heads=6, window_size=8, dropout=0.0).to(device)
            payload = torch.load(SARFUSIONFORMER_CHECKPOINT, map_location=device, weights_only=True)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
        elif name == "pix2pix":
            if not PIX2PIX_CHECKPOINT.is_file():
                raise SarTranslationError("CHECKPOINT_MISSING", "The Pix2Pix checkpoint is unavailable.")
            if str(WORKSPACE_ROOT) not in sys.path:
                sys.path.insert(0, str(WORKSPACE_ROOT))
            from src.pix2pix import Pix2Pix

            model = Pix2Pix(c_in=3, c_out=3, is_train=False).to(device)
            model.gen.load_state_dict(torch.load(PIX2PIX_CHECKPOINT, map_location=device, weights_only=True), strict=True)
            model.eval()
        else:  # pragma: no cover - guarded by configuration validation
            raise SarTranslationError("INVALID_MODEL", "The configured SAR translation model is unsupported.")
        self._models[name] = model
        self._load_count += 1
        return model

    def _load_color_corrector(self) -> Any:
        import torch

        if self._color_corrector is not None:
            return self._color_corrector
        if self._color_error and self._color_factory is None:
            raise SarTranslationError("COLOR_UNAVAILABLE", "The host could not load the optional color corrector.")
        if self._color_factory is not None:
            self._color_corrector = self._color_factory()
            return self._color_corrector
        if not COLOR_CORRECTOR_CHECKPOINT.is_file():
            raise SarTranslationError("COLOR_CHECKPOINT_MISSING", "The optional color corrector is unavailable.")

        class ColorCorrectionNet(torch.nn.Module):
            def __init__(self, hidden_channels: int = 32) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Conv2d(3, hidden_channels, 1), torch.nn.SiLU(inplace=True),
                    torch.nn.Conv2d(hidden_channels, hidden_channels, 1), torch.nn.SiLU(inplace=True),
                    torch.nn.Conv2d(hidden_channels, 3, 1),
                )

            def forward(self, image: Any) -> tuple[Any, Any]:
                correction = 0.10 * torch.tanh(self.net(image))
                return torch.clamp(image + correction, 0.0, 1.0), correction

        device = self._device or self._select_device()
        model = ColorCorrectionNet().to(device)
        payload = torch.load(COLOR_CORRECTOR_CHECKPOINT, map_location=device, weights_only=True)
        model.load_state_dict(payload["color_corrector"], strict=True)
        self._color_corrector = model.eval()
        return self._color_corrector

    def _ensure_model(self, name: str) -> tuple[Any, int, bool]:
        if not self.enabled:
            self._state = "disabled"
            raise SarTranslationError("DISABLED", "SAR-to-optical translated evidence is disabled.")
        if self._state == "failed":
            raise SarTranslationError("SERVICE_FAILED", "The SAR translation service remains unavailable until retry.")
        started = time.perf_counter()
        with self._load_lock:
            if name in self._models:
                self._state = "ready"
                self._reuse_count += 1
                return self._models[name], 0, True
            self._state = "loading"
            try:
                model = self._load_model(name)
                self._state = "ready"
                self._safe_error = None
                return model, _elapsed(started), False
            except SarTranslationError as error:
                self._model_errors[name] = error.message
                self._state = "unloaded"
                raise
            except Exception as error:
                self._model_errors[name] = "Model loading failed."
                self._state = "unloaded"
                raise SarTranslationError("LOAD_FAILED", f"The {name} translation model could not be loaded.") from error

    @staticmethod
    def _normalize_channel(channel: np.ndarray) -> np.ndarray:
        values = np.asarray(channel, dtype=np.float32)
        finite = np.isfinite(values)
        if not finite.any():
            raise SarTranslationError("NON_FINITE_INPUT", "The SAR input contains no finite values.")
        low, high = np.percentile(values[finite], [1, 99])
        output = np.zeros(values.shape, dtype=np.float32)
        if high > low:
            output[finite] = np.clip((values[finite] - low) / (high - low), 0, 1)
        else:
            output[finite] = 0.5
        return output

    def _run_sarfusionformer(self, model: Any, raster: np.ndarray) -> tuple[Any, Image.Image]:
        import torch
        import torch.nn.functional as functional
        from sarfusionformer import lab_to_rgb

        values = np.asarray(raster)
        channels = np.stack([self._normalize_channel(values[:, :, 0]), self._normalize_channel(values[:, :, 1])])
        tensor = functional.interpolate(torch.from_numpy(channels).unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).to(self._device, dtype=torch.float32)
        normalized_preview = Image.fromarray(np.round(channels.mean(axis=0).clip(0, 1) * 255).astype(np.uint8))
        with torch.inference_mode():
            lab = model(tensor)["lab"]
            rgb = lab_to_rgb(lab.float()).clamp(0, 1)
        if not torch.isfinite(rgb).all():
            raise SarTranslationError("NON_FINITE_OUTPUT", "SARFusionFormer produced a non-finite output.")
        return rgb, normalized_preview

    def _run_pix2pix(self, model: Any, model_image: Image.Image) -> tuple[Any, Image.Image]:
        import torch

        prepared = model_image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        array = np.asarray(prepared, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = ((tensor - 0.5) / 0.5).to(self._device, dtype=torch.float32)
        with torch.inference_mode():
            rgb = ((model(tensor) + 1.0) / 2.0).clamp(0, 1)
        if not torch.isfinite(rgb).all():
            raise SarTranslationError("NON_FINITE_OUTPUT", "Pix2Pix produced a non-finite output.")
        return rgb, prepared.convert("L")

    @staticmethod
    def _tensor_image(rgb: Any) -> tuple[Image.Image, list[float]]:
        array = rgb[0].detach().float().cpu().permute(1, 2, 0).numpy()
        if not np.isfinite(array).all():
            raise SarTranslationError("NON_FINITE_OUTPUT", "The translation output is non-finite.")
        array = np.clip(array, 0, 1)
        return Image.fromarray(np.round(array * 255).astype(np.uint8)), [float(array.min()), float(array.max())]

    def translate(
        self,
        raster: np.ndarray,
        metadata: ImageMetadata,
        model_image: Image.Image,
        *,
        automatic_preview: bool = False,
    ) -> SarTranslationResult:
        overall_started = time.perf_counter()
        if automatic_preview:
            is_automatic_sar_preview = (
                metadata.representation == RepresentationType.DISPLAY_PREVIEW
                and metadata.auto_detected_modality == ImageModality.SAR_PREVIEW
                and metadata.effective_modality == ImageModality.SAR_PREVIEW
                and metadata.user_confirmed_modality is None
            )
            if not is_automatic_sar_preview:
                raise SarTranslationError(
                    "INVALID_AUTOMATIC_PREVIEW",
                    "Automatic Pix2Pix activation is limited to conservatively auto-detected SAR previews.",
                )
            with self._load_lock:
                self._automatic_preview_active = True
                if self._state == "disabled":
                    self._state = "unloaded"
        if self._state == "failed":
            raise SarTranslationError("SERVICE_FAILED", "The SAR translation service remains unavailable until retry.")
        eligibility_started = time.perf_counter()
        eligibility = evaluate_translation_eligibility(
            raster,
            metadata,
            model_image,
            requested_model="pix2pix" if automatic_preview else None,
        )
        stages = {"sar_translation_eligibility": _elapsed(eligibility_started)}
        stage_statuses = {"sar_translation_eligibility": "success"}
        if not eligibility.eligible or not eligibility.selected_model:
            raise SarTranslationError("INELIGIBLE_INPUT", eligibility.reason or "The SAR input is not eligible for translation.")
        candidates = [eligibility.selected_model]
        alternate = "pix2pix" if eligibility.selected_model == "sarfusionformer" else "sarfusionformer"
        alternate_check = evaluate_translation_eligibility(raster, metadata, model_image, alternate)
        if alternate_check.eligible and alternate not in candidates:
            candidates.append(alternate)
        warnings: list[str] = []
        fallback_reason = eligibility.reason
        chosen: Optional[str] = None
        rgb: Any = None
        normalized_preview: Optional[Image.Image] = None
        model_reused = False
        load_ms = 0
        inference_ms = 0
        for index, name in enumerate(candidates):
            try:
                model, current_load_ms, reused = self._ensure_model(name)
                load_ms += current_load_ms
                model_reused = reused
                inference_started = time.perf_counter()
                with self._inference_lock:
                    if name == "sarfusionformer":
                        rgb, normalized_preview = self._run_sarfusionformer(model, raster)
                    else:
                        rgb, normalized_preview = self._run_pix2pix(model, model_image)
                inference_ms += _elapsed(inference_started)
                chosen = name
                if index:
                    warnings.append(f"The preferred translator was unavailable; {name} was used as the compatible fallback.")
                break
            except SarTranslationError as error:
                fallback_reason = error.message
                warnings.append(error.message)
        if chosen is None or rgb is None or normalized_preview is None:
            self._state = "failed"
            self._safe_error = "No compatible SAR translation model completed inference."
            raise SarTranslationError("TRANSLATION_UNAVAILABLE", self._safe_error)
        stages["sar_translation_model_load"] = load_ms
        stages["sar_translation_preprocessing"] = 0
        stages["sar_translation_inference"] = inference_ms
        stage_statuses.update({
            "sar_translation_model_load": "success",
            "sar_translation_preprocessing": "success",
            "sar_translation_inference": "success",
        })

        raw_output_image, raw_value_range = self._tensor_image(rgb)
        correction_used = False
        corrected_url: Optional[str] = None
        correction_started = time.perf_counter()
        if chosen == "sarfusionformer" and self.use_color_correction():
            try:
                corrector = self._load_color_corrector()
                import torch
                with self._inference_lock, torch.inference_mode():
                    corrected, _ = corrector(rgb)
                if not torch.isfinite(corrected).all():
                    raise SarTranslationError("NON_FINITE_COLOR_OUTPUT", "Color correction produced non-finite output.")
                rgb = corrected.clamp(0, 1)
                correction_used = True
            except Exception:
                warnings.append("Optional color correction failed safely; the uncorrected generated image was retained.")
        stages["sar_translation_color_correction"] = _elapsed(correction_started)
        stage_statuses["sar_translation_color_correction"] = "success" if correction_used or chosen != "sarfusionformer" or not self.use_color_correction() else "failed"

        if correction_used:
            output_image, value_range = self._tensor_image(rgb)
        else:
            output_image, value_range = raw_output_image, raw_value_range
        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        artifact_url: Optional[str] = None
        normalized_url: Optional[str] = None
        artifact_started = time.perf_counter()
        if save_artifacts_enabled():
            try:
                artifact_url = save_preview(raw_output_image)
            except ImageIngestionError:
                warnings.append("The generated optical-like evidence product could not be published safely.")
                artifact_url = None
                stage_statuses["sar_translation_artifact_generation"] = "failed"
            else:
                stage_statuses["sar_translation_artifact_generation"] = "success"
        else:
            stage_statuses["sar_translation_artifact_generation"] = "skipped"
        stages["sar_translation_artifact_generation"] = _elapsed(artifact_started)
        auxiliary_started = time.perf_counter()
        auxiliary_failed = False
        if save_artifacts_enabled():
            try:
                normalized_url = save_preview(normalized_preview)
            except ImageIngestionError:
                auxiliary_failed = True
                warnings.append("The optional normalized SAR translation preview could not be saved safely.")
            if correction_used:
                try:
                    corrected_url = save_preview(output_image)
                except ImageIngestionError:
                    auxiliary_failed = True
                    warnings.append("The optional color-corrected translation preview could not be saved safely.")
            stage_statuses["sar_translation_auxiliary_artifacts"] = "failed" if auxiliary_failed else "success"
        else:
            stage_statuses["sar_translation_auxiliary_artifacts"] = "skipped"
        stages["sar_translation_auxiliary_artifacts"] = _elapsed(auxiliary_started)
        if correction_used:
            raw_output_image.close()
        fallback_used = chosen != eligibility.requested_model
        checkpoint = SARFUSIONFORMER_CHECKPOINT if chosen == "sarfusionformer" else PIX2PIX_CHECKPOINT
        return SarTranslationResult(
            image=output_image,
            width=output_image.width,
            height=output_image.height,
            model_used="SARFusionFormer" if chosen == "sarfusionformer" else "Pix2Pix",
            fallback_used=fallback_used,
            fallback_reason=fallback_reason if fallback_used else None,
            color_correction_used=correction_used,
            device=self._device or "unknown",
            runtime_ms=_elapsed(overall_started),
            stage_durations_ms=stages,
            stage_statuses=stage_statuses,
            preprocessing_method=(
                "finite per-channel 1st/99th percentile normalization; channel order VV,VH; bilinear 256x256"
                if chosen == "sarfusionformer" else
                "ingestion SAR display converted to RGB; bicubic 256x256; [0,1] to [-1,1]"
            ),
            input_channel_interpretation=eligibility.channel_interpretation,
            output_value_range=value_range,
            warnings=list(dict.fromkeys(warnings)),
            provenance={
                "model": "SARFusionFormer" if chosen == "sarfusionformer" else "Pix2Pix",
                "checkpoint": checkpoint.name,
                "generated_representation": True,
                "model_reused": model_reused,
                "disclosure": DISCLOSURE,
            },
            artifact_url=artifact_url,
            normalized_sar_preview_url=normalized_url,
            color_corrected_artifact_url=corrected_url,
            content_hash=digest,
        )

    def retry(self) -> None:
        with self._load_lock:
            self._safe_error = None
            self._state = "unloaded" if self.enabled else "disabled"

    def reset_for_tests(self) -> None:
        with self._load_lock:
            self._models.clear()
            self._model_errors.clear()
            self._color_corrector = None
            self._color_error = None
            self._device = None
            self._safe_error = None
            self._load_count = 0
            self._reuse_count = 0
            self._automatic_preview_active = False
            self._state = "unloaded" if self.enabled else "disabled"


SAR_TRANSLATION_SERVICE = SarTranslationService()


def get_sar_translation_service() -> SarTranslationService:
    return SAR_TRANSLATION_SERVICE
