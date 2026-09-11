"""FastAPI inference service for independent SAR-to-optical models."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from starlette.datastructures import Headers

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

# Keep imports stable when the API is run from the repository root (Render),
# from this directory (local Uvicorn), or through an import-based test runner.
for import_path in (ROOT_DIR, APP_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from sarfusionformer import SARFusionFormer, lab_to_rgb
from src.pix2pix import Pix2Pix
from provider_settings import ProviderSettingsError, ProviderSettingsStore
from vision_analysis import ImageAnalysisService, VisionAnalysisError, VisionSettings
from satquery_agent import router as satquery_router
from satquery_agent.api import agent_image_query
from satquery_agent.comparison import (
    hash_bytes,
    record_model_result,
    store_model_preview,
    utc_now as comparison_utc_now,
)
from satquery_agent.models import (
    AgentResponse,
    ComparisonTask,
    ImageModality,
    InputMode,
    Modality,
    QuestionCategory,
)
from satquery_agent.entity_registry import (
    get_remote_sensing_entity,
    grounding_prompt_for,
    normalize_entity_text,
)
from satquery_agent.services.sve_service import SVECall, get_sve_service
from satquery_agent.services.sar_translation_service import get_sar_translation_service
from satquery_agent.specialists.grounder import GrounderError, get_grounder, normalize_grounding_target
from satquery_agent.specialists.rsvqa_specialist import (
    MODEL_USED as RSVQA_MODEL_USED,
    RSVQASpecialistError,
    get_rsvqa_specialist,
)
from satquery_agent.specialists.vqa import (
    QuestionIntent,
    get_vqa,
    is_groundable_count_target,
    normalize_benchmark_answer,
)
from satquery_agent.sve_artifacts import SVEError

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("sar-colorization")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(environment_key: str, default_relative_path: str) -> Path:
    """Resolve checkpoint locations consistently for local and hosted deployments.

    An environment value may be absolute (for a mounted disk) or relative to the
    repository root (for local use and the Render build download step). Defaults
    remain the original checkpoint locations used by this project.
    """
    configured = os.getenv(environment_key, default_relative_path).strip()
    candidate = Path(configured).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT_DIR / candidate).resolve()


PIX2PIX_CHECKPOINT = checkpoint_path("PIX2PIX_CHECKPOINT", "pix2pix_gen_180.pth")
SARFUSIONFORMER_CHECKPOINT = checkpoint_path(
    "SARFUSIONFORMER_CHECKPOINT", "models/checkpoints/sarfusionformer_256_decoder_best.pt"
)
COLOR_CORRECTOR_CHECKPOINT = checkpoint_path(
    "COLOR_CORRECTOR_CHECKPOINT", "models/checkpoints/color_corrector_256_best.pt"
)

app = FastAPI(title="SAR-to-Optical Colorization API", version="2.0.0")
app.include_router(satquery_router)


def load_optional_sve_on_startup() -> None:
    if os.getenv("SVE_LOAD_ON_STARTUP", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        get_sve_service().load()
    except SVEError as error:
        LOGGER.warning("Optional SatQuery Vision Encoder startup load was skipped: %s", error.public_message)


def load_rsvqa_specialist_on_startup() -> None:
    if os.getenv("RSVQA_LOAD_ON_STARTUP", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        get_rsvqa_specialist().load()
    except RSVQASpecialistError as error:
        LOGGER.warning("RSVQA Specialist v1 startup load failed; heuristic fallback remains available: %s", error)


def verify_grounding_specialist_on_startup() -> None:
    """Warm and smoke-verify the complete offline grounding pipeline for demo readiness."""
    if os.getenv("SATQUERY_GROUNDER_LOAD_ON_STARTUP", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        health = get_grounder().smoke_verify()
        if health.status != "ready" or not health.smoke_verified:
            LOGGER.warning("Grounding specialist startup smoke did not reach ready state: %s", health.error)
    except GrounderError as error:
        LOGGER.warning("Grounding specialist startup smoke failed [%s]: %s", error.code, error.message)


app.add_event_handler("startup", load_optional_sve_on_startup)
app.add_event_handler("startup", load_rsvqa_specialist_on_startup)
app.add_event_handler("startup", verify_grounding_specialist_on_startup)
CORS_ORIGINS = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS",
    "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:8520,http://localhost:8520",
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class ColorCorrectionNet(torch.nn.Module):
    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, hidden_channels, kernel_size=1),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

    def forward(self, coarse_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        correction = 0.10 * torch.tanh(self.net(coarse_rgb))
        return torch.clamp(coarse_rgb + correction, 0.0, 1.0), correction


def load_pix2pix() -> Pix2Pix:
    if not PIX2PIX_CHECKPOINT.is_file():
        raise FileNotFoundError("Pix2Pix checkpoint not found: {}".format(PIX2PIX_CHECKPOINT))
    model = Pix2Pix(c_in=3, c_out=3, is_train=False).to(DEVICE)
    model.gen.load_state_dict(
        torch.load(PIX2PIX_CHECKPOINT, map_location=DEVICE, weights_only=True),
        strict=True,
    )
    return model.eval()


def load_sarfusionformer() -> SARFusionFormer:
    if not SARFUSIONFORMER_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "SARFusionFormer checkpoint not found: {}".format(SARFUSIONFORMER_CHECKPOINT)
        )
    model = SARFusionFormer(
        input_channels=2,
        output_channels=3,
        base_channels=48,
        transformer_depth=4,
        attention_heads=6,
        window_size=8,
        dropout=0.0,
    ).to(DEVICE)
    checkpoint = torch.load(
        SARFUSIONFORMER_CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def load_color_corrector() -> ColorCorrectionNet:
    if not COLOR_CORRECTOR_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Color-corrector checkpoint not found: {}".format(COLOR_CORRECTOR_CHECKPOINT)
        )
    model = ColorCorrectionNet().to(DEVICE)
    checkpoint = torch.load(
        COLOR_CORRECTOR_CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["color_corrector"], strict=True)
    return model.eval()


def try_load(name: str, loader):
    try:
        model = loader()
        LOGGER.info("%s loaded successfully on %s", name, DEVICE)
        return model, None
    except Exception as error:
        LOGGER.warning("%s unavailable: %s", name, error)
        return None, str(error)


PIX2PIX_MODEL, PIX2PIX_ERROR = try_load("Pix2Pix", load_pix2pix)
SARFUSIONFORMER_MODEL, SARFUSIONFORMER_ERROR = try_load(
    "SARFusionFormer", load_sarfusionformer
)
COLOR_CORRECTOR, COLOR_CORRECTOR_ERROR = try_load(
    "Color corrector", load_color_corrector
)
get_sar_translation_service().register_external_models(
    pix2pix=PIX2PIX_MODEL,
    sarfusionformer=SARFUSIONFORMER_MODEL,
    color_corrector=COLOR_CORRECTOR,
    pix2pix_error=PIX2PIX_ERROR,
    sarfusionformer_error=SARFUSIONFORMER_ERROR,
    color_error=COLOR_CORRECTOR_ERROR,
    device=str(DEVICE),
)
VISION_ANALYSIS = ImageAnalysisService()
PROVIDER_SETTINGS = ProviderSettingsStore()
LATEST_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "pix2pix": {},
    "sarfusionformer": {},
}


def sync_vision_settings() -> None:
    """Refresh the analysis service after a key is changed in Settings."""
    credentials = PROVIDER_SETTINGS.credentials()
    VISION_ANALYSIS.settings = VisionSettings(
        provider=credentials.provider,
        api_key=credentials.api_key,
        model=credentials.model or ("gemini-2.5-flash" if credentials.provider == "gemini" else "gpt-4.1-mini"),
    )


sync_vision_settings()


class ProviderConfigurationRequest(BaseModel):
    provider: str
    api_key: str
    model: Optional[str] = None
    privacy_acknowledged: bool = False


class ImageAnalysisEndpointResponse(BaseModel):
    """Document the legacy image-review fields and additive VQA benchmark fields."""

    report: Dict[str, Any]
    provider: str
    model: str
    cached: bool
    request_id: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_details: Optional[Dict[str, Any]] = None
    caption: Optional[str] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    execution_trace: Optional[List[Dict[str, Any]]] = None
    task: Optional[str] = None
    model_used: Optional[str] = None
    processing_time_ms: Optional[int] = Field(default=None, ge=0)
    question: Optional[str] = None
    routed_question: Optional[str] = None
    status: Optional[str] = None
    result_status: Optional[str] = None
    warnings: Optional[List[str]] = None
    vqa_details: Optional[Dict[str, Any]] = None
    grounding_result: Optional[Dict[str, Any]] = None
    sve_result: Optional[Dict[str, Any]] = None


def model_status(model: Optional[torch.nn.Module], error: Optional[str], checkpoint: Path) -> Dict[str, Any]:
    return {
        "available": model is not None,
        "checkpoint": checkpoint.name,
        "error": error,
    }


def require_model(model: Optional[torch.nn.Module], error: Optional[str], name: str):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="{} is unavailable. {}".format(name, error or "Check the checkpoint path."),
        )
    return model


def read_upload(upload: UploadFile) -> bytes:
    contents = upload.file.read()
    if not contents:
        raise ValueError("The uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Files must be 20 MB or smaller.")
    return contents


def decode_rgb(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Upload a valid PNG, JPEG, or TIFF image.") from error
    return image.convert("RGB")


def decode_grayscale(image_bytes: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename or "").suffix.lower()
    try:
        if suffix == ".npy":
            array = np.load(io.BytesIO(image_bytes), allow_pickle=False)
        else:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            array = np.asarray(image)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("Upload a valid .npy, TIFF, PNG, or JPEG single-channel image.") from error

    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("VV and VH inputs must be single-channel two-dimensional arrays.")
    return array.astype(np.float32, copy=False)


def load_combined_sar_npy(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, str, Tuple[int, ...]]:
    """Load a two-channel VV/VH NumPy array in channel-first or channel-last form."""
    try:
        array = np.asarray(np.load(io.BytesIO(image_bytes), allow_pickle=False))
    except (ValueError, OSError) as error:
        raise ValueError("Upload a valid combined VV/VH NumPy (.npy) file.") from error

    detected_shape = tuple(array.shape)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise ValueError("Combined SAR arrays must contain numeric, non-object values.")
    if np.iscomplexobj(array):
        raise ValueError("Combined SAR arrays must contain real-valued VV and VH channels.")
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            "Expected a 3D two-channel SAR array, got shape {}".format(detected_shape)
        )

    is_chw = array.shape[0] == 2
    is_hwc = array.shape[-1] == 2
    if is_chw and is_hwc:
        raise ValueError(
            "Ambiguous two-channel SAR layout in shape {}. Use an unambiguous CHW or HWC array."
            .format(detected_shape)
        )
    if is_chw:
        sar = array
        layout = "CHW"
    elif is_hwc:
        sar = np.moveaxis(array, -1, 0)
        layout = "HWC"
    else:
        raise ValueError(
            "Could not find exactly two SAR channels in shape {}".format(detected_shape)
        )

    sar = sar.astype(np.float32, copy=False)
    if not np.isfinite(sar).any():
        raise ValueError("Combined SAR input contains no finite values.")
    return sar[0], sar[1], layout, detected_shape


def normalize_channel(array: np.ndarray) -> np.ndarray:
    """Apply the exact per-channel percentile normalization used in training."""
    low, high = np.percentile(array, [1, 99])
    if high - low <= np.finfo(np.float32).eps:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return rgb_float_to_png(array)


def rgb_float_to_png(rgb: np.ndarray) -> Image.Image:
    """Encode an RGB float image without changing its radiometric values first."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB output must have shape [height, width, 3].")
    encoded = (np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8)
    return Image.fromarray(encoded, mode="RGB")


def contrast_stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    """Global display-only 2nd/98th percentile stretch that preserves RGB balance."""
    rgb = np.asarray(rgb, dtype=np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)

    low = float(np.percentile(rgb, 2))
    high = float(np.percentile(rgb, 98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return rgb.copy()

    stretched = (rgb - low) / (high - low)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def array_diagnostics(array: np.ndarray, prefix: str) -> Dict[str, Any]:
    array = np.asarray(array, dtype=np.float32)
    return {
        "{}_min".format(prefix): float(array.min()),
        "{}_max".format(prefix): float(array.max()),
        "{}_mean".format(prefix): float(array.mean()),
        "{}_std".format(prefix): float(array.std()),
    }


def rgb_diagnostics(rgb: np.ndarray, prefix: str) -> Dict[str, Any]:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB diagnostics require a [height, width, 3] array.")
    return {
        **array_diagnostics(rgb, prefix),
        "{}_channel_mean".format(prefix): [float(value) for value in rgb.mean(axis=(0, 1))],
        "{}_channel_std".format(prefix): [float(value) for value in rgb.std(axis=(0, 1))],
    }


def display_representation(rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any], Optional[str]]:
    """Return a visual-only representation and diagnostics for one raw RGB image."""
    rgb = np.asarray(rgb, dtype=np.float32)
    finite_rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    low = float(np.percentile(finite_rgb, 2))
    high = float(np.percentile(finite_rgb, 98))
    raw_std = float(finite_rgb.std())
    warning = None
    if raw_std < 0.001:
        # Do not turn a nearly constant scientific result into apparent detail.
        display_rgb = finite_rgb.copy()
        warning = (
            "Raw RGB standard deviation is below 0.001; contrast enhancement was skipped. "
            "Review the model input and checkpoint rather than interpreting amplified noise."
        )
    else:
        display_rgb = contrast_stretch_rgb(finite_rgb)
    diagnostics = {
        **rgb_diagnostics(finite_rgb, "raw_rgb"),
        **rgb_diagnostics(display_rgb, "display_rgb"),
        "stretch_low": low,
        "stretch_high": high,
        "stretch_applied": warning is None and high > low,
    }
    return display_rgb, diagnostics, warning


def channel_preview(channel: np.ndarray) -> Image.Image:
    return Image.fromarray((channel.clip(0, 1) * 255).round().astype(np.uint8), mode="L")


def prepare_sarfusionformer_input(
    vv_array: np.ndarray, vh_array: np.ndarray
) -> Tuple[torch.Tensor, Dict[str, Image.Image]]:
    """Produce the exact [1, 2, 256, 256] float32 tensor used during training."""
    if vv_array.shape != vh_array.shape:
        raise ValueError("VV and VH inputs must have matching spatial dimensions.")
    if vv_array.ndim != 2:
        raise ValueError("VV and VH inputs must be two-dimensional SAR channels.")

    # Keep this order aligned with the training pipeline: stack, replace invalid
    # values, normalize each channel independently, then resize to model resolution.
    source_sar = np.stack([vv_array, vh_array], axis=0).astype(np.float32, copy=False)
    if not np.isfinite(source_sar).any():
        raise ValueError("SAR array contains no finite values.")
    source_sar = np.nan_to_num(source_sar, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_sar = np.stack(
        [normalize_channel(source_sar[0]), normalize_channel(source_sar[1])], axis=0
    ).astype(np.float32, copy=False)

    sar_tensor = torch.from_numpy(normalized_sar).unsqueeze(0)
    sar_tensor = F.interpolate(
        sar_tensor, size=(256, 256), mode="bilinear", align_corners=False
    ).to(dtype=torch.float32)
    preview_sar = sar_tensor[0].numpy()

    # SAR is not an RGB image.  A neutral intensity composite avoids falsely
    # presenting VV/VH as red/green/blue colours.
    combined_preview = channel_preview(preview_sar.mean(axis=0)).convert("RGB")
    return sar_tensor.to(DEVICE, dtype=torch.float32), {
        "vv": channel_preview(preview_sar[0]),
        "vh": channel_preview(preview_sar[1]),
        "sar": combined_preview,
    }


def calculate_metrics(
    prediction: Union[Image.Image, np.ndarray], ground_truth: Image.Image
) -> Dict[str, Optional[float]]:
    """Calculate metrics from raw radiometric RGB, never display-enhanced RGB."""
    if isinstance(prediction, Image.Image):
        prediction_array = np.asarray(prediction.convert("RGB"), dtype=np.float32) / 255.0
    else:
        prediction_array = np.asarray(prediction, dtype=np.float32)
        if prediction_array.ndim != 3 or prediction_array.shape[-1] != 3:
            raise ValueError("Metric prediction must have shape [height, width, 3].")
        prediction_array = np.nan_to_num(
            prediction_array, nan=0.0, posinf=1.0, neginf=0.0
        ).clip(0.0, 1.0)
    target = ground_truth.convert("RGB").resize(
        (prediction_array.shape[1], prediction_array.shape[0]), Image.Resampling.BICUBIC
    )
    target_array = np.asarray(target, dtype=np.float32) / 255.0
    psnr = peak_signal_noise_ratio(target_array, prediction_array, data_range=1.0)
    return {
        "psnr": None if not np.isfinite(psnr) else float(psnr),
        "ssim": float(
            structural_similarity(target_array, prediction_array, channel_axis=2, data_range=1.0)
        ),
        "rgb_l1": float(np.mean(np.abs(target_array - prediction_array))),
    }


def pix2pix_generate(image: Image.Image) -> Tuple[Image.Image, float]:
    model = require_model(PIX2PIX_MODEL, PIX2PIX_ERROR, "Pix2Pix")
    resized = image.resize((256, 256), Image.Resampling.BICUBIC)
    input_array = np.asarray(resized, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(input_array).permute(2, 0, 1)
    input_tensor = ((input_tensor - 0.5) / 0.5).unsqueeze(0).to(DEVICE)
    LOGGER.info("Pix2Pix input shape: %s", tuple(input_tensor.shape))
    start = time.perf_counter()
    with torch.inference_mode():
        output = ((model(input_tensor)[0] + 1.0) / 2.0).clamp(0, 1)
    duration_ms = (time.perf_counter() - start) * 1000
    LOGGER.info("Pix2Pix inference duration: %.2f ms", duration_ms)
    return tensor_to_image(output), duration_ms


def sarfusionformer_generate(
    vv_array: np.ndarray, vh_array: np.ndarray, apply_color_correction: bool = False
) -> Dict[str, Any]:
    model = require_model(
        SARFUSIONFORMER_MODEL, SARFUSIONFORMER_ERROR, "SARFusionFormer"
    )
    sar, previews = prepare_sarfusionformer_input(vv_array, vh_array)
    LOGGER.info(
        "SARFusionFormer input shape=%s dtype=%s range=[%.6f, %.6f]",
        tuple(sar.shape),
        sar.dtype,
        sar.amin().item(),
        sar.amax().item(),
    )
    start = time.perf_counter()
    with torch.inference_mode():
        prediction_lab = model(sar)["lab"]
        if not torch.isfinite(prediction_lab).all():
            raise ValueError("SARFusionFormer produced non-finite LAB values.")
        raw_rgb = lab_to_rgb(prediction_lab.float()).clamp(0, 1)
        if not torch.isfinite(raw_rgb).all():
            raise ValueError("LAB-to-RGB conversion produced non-finite RGB values.")

        # This is the scientific prediction. The display path receives an
        # independent copy and is checked below so it can never overwrite it.
        raw_rgb_hwc = raw_rgb[0].detach().cpu().permute(1, 2, 0).numpy().copy()
        raw_rgb_before_display = raw_rgb_hwc.copy()
        display_rgb_hwc, diagnostics, warning = display_representation(raw_rgb_hwc.copy())
        if not np.array_equal(raw_rgb_hwc, raw_rgb_before_display):
            raise RuntimeError("Display processing modified the raw RGB prediction.")
        diagnostics.update(
            array_diagnostics(prediction_lab[0].detach().cpu().numpy(), "prediction_lab")
        )
        diagnostics.update(
            {
                "prediction_shape": list(raw_rgb.shape),
                "prediction_dtype": str(raw_rgb.dtype),
                "lab_finite": True,
                "rgb_finite": True,
                "raw_prediction_preserved": True,
                "inference_successful": True,
            }
        )
        if diagnostics["raw_rgb_max"] - diagnostics["raw_rgb_min"] < 0.1:
            narrow_range_message = (
                "Model inference succeeded. The prediction has a narrow radiometric range, "
                "so the raw image appears dark."
            )
            warning = (
                narrow_range_message if warning is None else "{} {}".format(warning, narrow_range_message)
            )
        LOGGER.info("Loaded checkpoint: %s", SARFUSIONFORMER_CHECKPOINT.name)
        LOGGER.info("Prediction shape: %s", tuple(raw_rgb.shape))
        LOGGER.info("Prediction dtype: %s", raw_rgb.dtype)
        LOGGER.info(
            "Prediction min/max: %.6f / %.6f; Prediction mean/std: %.6f / %.6f",
            diagnostics["raw_rgb_min"],
            diagnostics["raw_rgb_max"],
            diagnostics["raw_rgb_mean"],
            diagnostics["raw_rgb_std"],
        )
        LOGGER.info(
            "RGB image range: [%.6f, %.6f]; Enhanced image range: [%.6f, %.6f]",
            diagnostics["raw_rgb_min"],
            diagnostics["raw_rgb_max"],
            diagnostics["display_rgb_min"],
            diagnostics["display_rgb_max"],
        )
        LOGGER.info("Lab-to-RGB conversion: completed; Inference successful: True")
        LOGGER.info(
            "SARFusionFormer Lab min=%.6f max=%.6f mean=%.6f std=%.6f; "
            "raw RGB min=%.6f max=%.6f mean=%.6f std=%.6f; stretch=[%.6f, %.6f]",
            diagnostics["prediction_lab_min"],
            diagnostics["prediction_lab_max"],
            diagnostics["prediction_lab_mean"],
            diagnostics["prediction_lab_std"],
            diagnostics["raw_rgb_min"],
            diagnostics["raw_rgb_max"],
            diagnostics["raw_rgb_mean"],
            diagnostics["raw_rgb_std"],
            diagnostics["stretch_low"],
            diagnostics["stretch_high"],
        )
        corrected_rgb = None
        corrected_raw_hwc = None
        corrected_display_hwc = None
        corrected_diagnostics = None
        if apply_color_correction and COLOR_CORRECTOR is not None:
            corrected_rgb, _ = COLOR_CORRECTOR(raw_rgb)
            corrected_raw_hwc = corrected_rgb[0].detach().cpu().permute(1, 2, 0).numpy()
            corrected_display_hwc, corrected_diagnostics, corrected_warning = display_representation(
                corrected_raw_hwc
            )
            if corrected_warning:
                warning = corrected_warning if warning is None else "{} {}".format(warning, corrected_warning)
        elif apply_color_correction:
            correction_warning = COLOR_CORRECTOR_ERROR or "Color corrector is unavailable."
            warning = correction_warning if warning is None else "{} {}".format(warning, correction_warning)
    duration_ms = (time.perf_counter() - start) * 1000
    LOGGER.info("SARFusionFormer inference duration: %.2f ms", duration_ms)

    return {
        "raw_rgb": raw_rgb_hwc,
        "display_rgb": display_rgb_hwc,
        "corrected_raw_rgb": corrected_raw_hwc,
        "corrected_display_rgb": corrected_display_hwc,
        "previews": previews,
        "duration_ms": duration_ms,
        "warning": warning,
        "diagnostics": diagnostics,
        "corrected_diagnostics": corrected_diagnostics,
    }


def pix2pix_payload(
    input_bytes: bytes, ground_truth_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    source = decode_rgb(input_bytes)
    output, duration_ms = pix2pix_generate(source)
    target = decode_rgb(ground_truth_bytes) if ground_truth_bytes else None
    payload = {
        "input_preview": image_to_base64(source),
        "output": image_to_base64(output),
        "metrics": calculate_metrics(output, target) if target else None,
        "inference_time_ms": round(duration_ms, 2),
        "checkpoint": PIX2PIX_CHECKPOINT.name,
    }
    LATEST_BENCHMARKS["pix2pix"] = {
        "metrics": payload["metrics"],
        "inference_time_ms": payload["inference_time_ms"],
        "sample": "current_sample",
    }
    return payload


def checkpoint_size_mb(checkpoint: Path) -> Optional[float]:
    if not checkpoint.is_file():
        return None
    return round(checkpoint.stat().st_size / (1024 * 1024), 2)


def benchmark_model_payload(name: str, checkpoint: Path) -> Dict[str, Any]:
    latest = LATEST_BENCHMARKS[name]
    metrics = latest.get("metrics") or {}
    return {
        "metrics": {
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "rgb_l1": metrics.get("rgb_l1"),
            "inference_time_ms": latest.get("inference_time_ms"),
            "model_size_mb": checkpoint_size_mb(checkpoint),
            "gpu_memory_mb": latest.get("gpu_memory_mb"),
        },
        "sample": latest.get("sample"),
        "checkpoint": checkpoint.name,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ready",
        "device": str(DEVICE),
        "models": {
            "pix2pix": model_status(PIX2PIX_MODEL, PIX2PIX_ERROR, PIX2PIX_CHECKPOINT),
            "sarfusionformer": model_status(
                SARFUSIONFORMER_MODEL, SARFUSIONFORMER_ERROR, SARFUSIONFORMER_CHECKPOINT
            ),
            "color_corrector": model_status(
                COLOR_CORRECTOR, COLOR_CORRECTOR_ERROR, COLOR_CORRECTOR_CHECKPOINT
            ),
            "sar_translation_service": get_sar_translation_service().health_payload(),
        },
        "vision_analysis": {**VISION_ANALYSIS.status(), **PROVIDER_SETTINGS.public_status()},
    }


@app.get("/api/benchmark")
def benchmark() -> Dict[str, Any]:
    """Return real checkpoint metadata and the latest in-memory measured sample."""
    return {
        "models": {
            "pix2pix": benchmark_model_payload("pix2pix", PIX2PIX_CHECKPOINT),
            "sarfusionformer": benchmark_model_payload("sarfusionformer", SARFUSIONFORMER_CHECKPOINT),
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> StreamingResponse:
    """Legacy Pix2Pix endpoint retained for existing clients."""
    try:
        payload = pix2pix_payload(read_upload(file))
        output_bytes = base64.b64decode(payload["output"])
        return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/evaluate")
async def evaluate(
    sar_file: UploadFile = File(...), optical_file: UploadFile = File(...)
) -> StreamingResponse:
    """Legacy Pix2Pix evaluation endpoint retained for existing clients."""
    try:
        payload = pix2pix_payload(read_upload(sar_file), read_upload(optical_file))
        output_bytes = base64.b64decode(payload["output"])
        metrics = payload["metrics"] or {}
        return StreamingResponse(
            io.BytesIO(output_bytes),
            media_type="image/png",
            headers={
                "X-PSNR": "{:.2f}".format(metrics["psnr"]) if metrics["psnr"] is not None else "N/A",
                "X-SSIM": "{:.4f}".format(metrics["ssim"]) if metrics["ssim"] is not None else "N/A",
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/pix2pix/infer")
async def pix2pix_infer(
    file: UploadFile = File(...), ground_truth: Optional[UploadFile] = File(None)
) -> JSONResponse:
    try:
        target = read_upload(ground_truth) if ground_truth else None
        input_bytes = read_upload(file)
        payload = pix2pix_payload(input_bytes, target)
        request_id = str(uuid.uuid4())
        input_image = decode_rgb(input_bytes)
        output_bytes = base64.b64decode(payload["output"])
        output_image = decode_rgb(output_bytes)
        input_preview = store_model_preview(input_image, "SAR input", "input", Modality.SAR)
        output_preview = store_model_preview(output_image, "Pix2Pix optical reconstruction", "output", Modality.OPTICAL)
        record_model_result(
            task=ComparisonTask.PIX2PIX,
            request_id=request_id,
            primary_hash=hash_bytes(input_bytes),
            component_hashes=(),
            input_previews=[input_preview],
            output_previews=[output_preview],
            output_hashes=[hash_bytes(output_bytes)],
            statistics={"reconstruction": payload["metrics"] or {}},
            execution_duration_ms=payload["inference_time_ms"],
            device=str(DEVICE),
            provenance={"model": "Pix2Pix", "checkpoint": PIX2PIX_CHECKPOINT.name, "method": "SAR-to-optical reconstruction"},
            warnings=[],
            safe_parameters={"ground_truth_available": target is not None},
        )
        payload.update({"request_id": request_id, "created_at": comparison_utc_now()})
        return JSONResponse(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/sarfusionformer/infer")
async def sarfusionformer_infer(
    combined_file: Optional[UploadFile] = File(None),
    vv_file: Optional[UploadFile] = File(None),
    vh_file: Optional[UploadFile] = File(None),
    ground_truth: Optional[UploadFile] = File(None),
    apply_color_correction: bool = Form(False),
) -> JSONResponse:
    try:
        primary_hash: str
        component_hashes: list[str]
        source_mode: str
        if combined_file is not None:
            if Path(combined_file.filename or "").suffix.lower() != ".npy":
                raise ValueError("Combined VV/VH input must be a NumPy (.npy) file.")
            combined_bytes = read_upload(combined_file)
            vv, vh, channel_layout, detected_shape = load_combined_sar_npy(
                combined_bytes
            )
            primary_hash = hash_bytes(combined_bytes)
            component_hashes = []
            source_mode = "combined_vv_vh"
        elif vv_file is not None and vh_file is not None:
            vv_bytes = read_upload(vv_file)
            vh_bytes = read_upload(vh_file)
            vv = decode_grayscale(vv_bytes, vv_file.filename or "")
            vh = decode_grayscale(vh_bytes, vh_file.filename or "")
            channel_layout = "Separate VV and VH files"
            detected_shape = tuple(vv.shape)
            primary_hash = hash_bytes(vv_bytes)
            component_hashes = [hash_bytes(vv_bytes), hash_bytes(vh_bytes)]
            source_mode = "separate_vv_vh"
        elif vv_file is None:
            raise ValueError("Provide a combined VV/VH .npy file or upload a VV file.")
        else:
            raise ValueError("Provide a combined VV/VH .npy file or upload a VH file.")
        result = sarfusionformer_generate(
            vv, vh, apply_color_correction=apply_color_correction
        )
        target_bytes = read_upload(ground_truth) if ground_truth else None
        target = decode_rgb(target_bytes) if target_bytes else None
        raw_metrics = calculate_metrics(result["raw_rgb"], target) if target else None
        corrected_metrics = (
            calculate_metrics(result["corrected_raw_rgb"], target)
            if target and result["corrected_raw_rgb"] is not None
            else None
        )
        LATEST_BENCHMARKS["sarfusionformer"] = {
            "metrics": raw_metrics,
            "inference_time_ms": round(result["duration_ms"], 2),
            "sample": "current_sample",
        }
        payload = {
                "raw_output": image_to_base64(rgb_float_to_png(result["raw_rgb"])),
                "display_output": image_to_base64(rgb_float_to_png(result["display_rgb"])),
                "corrected_raw_output": (
                    image_to_base64(rgb_float_to_png(result["corrected_raw_rgb"]))
                    if result["corrected_raw_rgb"] is not None
                    else None
                ),
                "corrected_display_output": (
                    image_to_base64(rgb_float_to_png(result["corrected_display_rgb"]))
                    if result["corrected_display_rgb"] is not None
                    else None
                ),
                "vv_preview": image_to_base64(result["previews"]["vv"]),
                "vh_preview": image_to_base64(result["previews"]["vh"]),
                "sar_preview": image_to_base64(result["previews"]["sar"]),
                "metrics_raw": raw_metrics,
                "metrics_corrected": corrected_metrics,
                "inference_time_ms": round(result["duration_ms"], 2),
                "detected_shape": list(detected_shape),
                "channel_layout": channel_layout,
                "checkpoint": SARFUSIONFORMER_CHECKPOINT.name,
                "color_checkpoint": (
                    COLOR_CORRECTOR_CHECKPOINT.name
                    if result["corrected_raw_rgb"] is not None
                    else None
                ),
                "warning": result["warning"],
                "diagnostics": result["diagnostics"],
                "corrected_diagnostics": result["corrected_diagnostics"],
            }
        request_id = str(uuid.uuid4())
        input_previews = [
            store_model_preview(result["previews"]["vv"], "VV input", "input", Modality.SAR),
            store_model_preview(result["previews"]["vh"], "VH input", "input", Modality.SAR),
            store_model_preview(result["previews"]["sar"], "Combined SAR input", "input", Modality.SAR),
        ]
        raw_image = rgb_float_to_png(result["raw_rgb"])
        display_image = rgb_float_to_png(result["display_rgb"])
        if get_sve_service().enabled:
            raw_buffer = io.BytesIO()
            raw_image.save(raw_buffer, format="PNG")
            generated_hash = hash_bytes(raw_buffer.getvalue())
            if target is not None and target_bytes is not None:
                sve_call = await run_in_threadpool(
                    get_sve_service().compare,
                    target,
                    hash_bytes(target_bytes),
                    raw_image,
                    generated_hash,
                    label="Optical-to-generated-RGB semantic consistency",
                    disclaimer=(
                        "This is supporting scene-level semantic evidence only; it does not establish sensor-native "
                        "equivalence, physical accuracy, or registration accuracy."
                    ),
                )
            else:
                sve_call = await run_in_threadpool(
                    get_sve_service().analyze,
                    raw_image,
                    generated_hash,
                )
                sve_call = SVECall(
                    result=sve_call.result.model_copy(update={
                        "warning": "Scene priors were computed from SARFusionFormer-generated RGB-like imagery, not raw SAR or reference optical imagery."
                    }),
                    trace=sve_call.trace,
                )
            payload["sve_result"] = sve_call.result.model_dump(mode="json")
        else:
            payload["sve_result"] = None
        output_previews = [
            store_model_preview(raw_image, "Raw optical reconstruction", "output", Modality.OPTICAL),
            store_model_preview(display_image, "Enhanced display reconstruction", "output", Modality.OPTICAL),
        ]
        if result["corrected_raw_rgb"] is not None:
            output_previews.append(store_model_preview(rgb_float_to_png(result["corrected_raw_rgb"]), "Color-corrected raw reconstruction", "output", Modality.OPTICAL))
        encoded_outputs = [base64.b64decode(payload["raw_output"]), base64.b64decode(payload["display_output"])]
        if payload["corrected_raw_output"]:
            encoded_outputs.append(base64.b64decode(payload["corrected_raw_output"]))
        record_model_result(
            task=ComparisonTask.SARFUSIONFORMER,
            request_id=request_id,
            primary_hash=primary_hash,
            component_hashes=component_hashes,
            input_previews=input_previews,
            output_previews=output_previews,
            output_hashes=[hash_bytes(value) for value in encoded_outputs],
            statistics={"raw_metrics": raw_metrics or {}, "corrected_metrics": corrected_metrics or {}, "detected_shape": list(detected_shape), "sve_result": payload.get("sve_result") or {}},
            execution_duration_ms=result["duration_ms"],
            device=str(DEVICE),
            provenance={"model": "SARFusionFormer", "checkpoint": SARFUSIONFORMER_CHECKPOINT.name, "color_checkpoint": COLOR_CORRECTOR_CHECKPOINT.name if result["corrected_raw_rgb"] is not None else None, "method": "structure-preserving SAR-to-optical reconstruction"},
            warnings=[result["warning"]] if result["warning"] else [],
            safe_parameters={"source_mode": source_mode, "color_correction_requested": apply_color_correction, "ground_truth_available": target is not None},
        )
        payload.update({"request_id": request_id, "created_at": comparison_utc_now()})
        return JSONResponse(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/compare")
async def compare(
    pix2pix_output: UploadFile = File(...),
    sarfusionformer_output: UploadFile = File(...),
    ground_truth: UploadFile = File(...),
) -> JSONResponse:
    try:
        target = decode_rgb(read_upload(ground_truth))
        pix_metrics = calculate_metrics(decode_rgb(read_upload(pix2pix_output)), target)
        sar_metrics = calculate_metrics(decode_rgb(read_upload(sarfusionformer_output)), target)
        LATEST_BENCHMARKS["pix2pix"] = {"metrics": pix_metrics, "sample": "current_comparison"}
        LATEST_BENCHMARKS["sarfusionformer"] = {"metrics": sar_metrics, "sample": "current_comparison"}
        return JSONResponse({"pix2pix": pix_metrics, "sarfusionformer": sar_metrics})
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _agent_upload(image_bytes: bytes) -> UploadFile:
    """Create an in-memory upload for the existing SatQuery multipart pipeline."""
    return UploadFile(
        file=io.BytesIO(image_bytes),
        filename="rsvqa-image.png",
        headers=Headers({"content-type": "image/png"}),
    )


def _rsvqa_routed_question(question: str) -> str:
    """Map common RSVQA wording onto the existing controlled VQA taxonomy."""
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    presence_terms = ("is there", "visible", "present", "contain", "show", "can you see")
    if "water" not in normalized and any(term in normalized for term in ("river", "lake", "reservoir")):
        if any(term in normalized for term in presence_terms):
            return "Is a water body visible?"
    if "urban" in normalized and normalized.startswith(("is this urban", "is the scene urban")):
        return "Is this mainly urban or rural?"
    return question.strip()


def _count_grounding_query(question: str) -> Optional[Tuple[str, str]]:
    """Convert a controlled count target into a validated localization request."""
    intent = get_vqa().classify_question(question)
    if intent.category != QuestionCategory.COUNT_VQA or not is_groundable_count_target(intent.target):
        return None
    candidate = grounding_prompt_for(intent.raw_target or intent.target or "")
    if candidate is None:
        return None
    try:
        target = normalize_grounding_target(f"Locate the {candidate}.")
    except GrounderError:
        return None
    return f"Locate the {target}.", target


def _grounding_query_for_concept(
    target: Optional[str],
    raw_target: Optional[str] = None,
    *,
    require_count: bool = True,
) -> Optional[Tuple[str, str]]:
    """Build a Grounding DINO request only for a supported controlled concept."""
    entity = get_remote_sensing_entity(target)
    if entity is None or not entity.grounding_prompts or (require_count and not entity.count_meaningful):
        return None
    prompt = grounding_prompt_for(raw_target or target or "")
    if prompt is None:
        return None
    try:
        canonical = normalize_grounding_target(f"Locate the {prompt}.")
    except GrounderError:
        return None
    return f"Locate the {canonical}.", canonical


async def _agent_single_image_query(image_bytes: bytes, query: str) -> AgentResponse:
    """Reuse the authoritative SatQuery single-image execution entry point."""
    return await agent_image_query(
        query=query,
        input_mode=InputMode.SINGLE,
        primary_modality=Modality.OPTICAL,
        primary_image_modality=ImageModality.OPTICAL_RGB,
        primary_image=_agent_upload(image_bytes),
        secondary_modality=None,
        secondary_image=None,
        primary_date=None,
        secondary_date=None,
        use_cache=False,
        force_rerun=True,
    )


async def _run_satquery_vqa(
    image_bytes: bytes,
    question: str,
) -> Tuple[AgentResponse, AgentResponse, Dict[str, AgentResponse], str, QuestionIntent]:
    """Reuse caption/VQA execution and add accepted-region support when required."""
    intent = get_vqa().classify_question(question)
    benchmark_categories = {
        QuestionCategory.RURAL_URBAN_CLASSIFICATION,
        QuestionCategory.PRESENCE_VQA,
        QuestionCategory.COUNT_VQA,
        QuestionCategory.COMPARISON_VQA,
    }
    # The existing alias helper remains available to legacy callers, but the
    # controlled parser must see the original wording to retain the exact RSVQA
    # family and entity information.
    routed_question = question.strip() if intent.category in benchmark_categories else _rsvqa_routed_question(question)
    caption_response = await _agent_single_image_query(image_bytes, "Describe this image")
    response = await _agent_single_image_query(image_bytes, routed_question)
    grounding_responses: Dict[str, AgentResponse] = {}
    targets: list[Tuple[str, Optional[str], Optional[str]]] = []
    if intent.category == QuestionCategory.PRESENCE_VQA:
        targets = [("target", intent.target, intent.raw_target)]
    elif intent.category == QuestionCategory.COUNT_VQA and intent.spatial_relation is None:
        targets = [("target", intent.target, intent.raw_target)]
    elif intent.category == QuestionCategory.COMPARISON_VQA:
        targets = [
            ("target", intent.target, intent.raw_target),
            ("secondary_target", intent.secondary_target, intent.raw_secondary_target),
        ]
    response_by_query: Dict[str, AgentResponse] = {}
    for key, target, raw_target in targets:
        grounding_request = _grounding_query_for_concept(
            target,
            raw_target,
            require_count=intent.category != QuestionCategory.PRESENCE_VQA,
        )
        if grounding_request is None:
            continue
        grounding_query, _ = grounding_request
        if grounding_query not in response_by_query:
            response_by_query[grounding_query] = await _agent_single_image_query(image_bytes, grounding_query)
        grounding_responses[key] = response_by_query[grounding_query]
    return caption_response, response, grounding_responses, routed_question, intent


_BENCHMARK_VQA_CATEGORIES = {
    QuestionCategory.RURAL_URBAN_CLASSIFICATION,
    QuestionCategory.PRESENCE_VQA,
    QuestionCategory.COUNT_VQA,
    QuestionCategory.COMPARISON_VQA,
}
_COUNT_LIMITATION = (
    "Count reflects accepted localized regions, not calibrated object totals. "
    "Grounding DINO regions are supporting localization evidence, not ground truth."
)


def _grounding_region_total(response: Optional[AgentResponse]) -> Optional[int]:
    if response is None or response.grounding_result is None:
        return None
    return len(response.grounding_result.detections)


def _accepted_region_count(response: Optional[AgentResponse]) -> Optional[int]:
    """Return only positive accepted-region evidence; zero is unconfirmed."""
    total = _grounding_region_total(response)
    return total if total is not None and total > 0 else None


@dataclass(frozen=True)
class CountEvidence:
    count: Optional[int]
    reason_code: str
    evidence_source: Optional[str]
    accepted_region_total: Optional[int]


@dataclass(frozen=True)
class BenchmarkVQADecision:
    answer: Optional[str]
    confidence_response: AgentResponse
    limitation: Optional[str]
    answer_source: Optional[str]
    reason_code: str
    details: Dict[str, Any]
    trace_stage: str


def _count_evidence(
    target: Optional[str],
    response: Optional[AgentResponse],
    *,
    spatial_relation: Optional[str] = None,
) -> CountEvidence:
    entity = get_remote_sensing_entity(target)
    if spatial_relation:
        return CountEvidence(None, "spatial_relation_unsupported", None, _grounding_region_total(response))
    if entity is None or not entity.count_meaningful or not entity.grounding_prompts:
        return CountEvidence(None, "unsupported_count_target", None, _grounding_region_total(response))
    total = _grounding_region_total(response)
    if total is None:
        return CountEvidence(None, "grounding_unavailable", None, None)
    if total == 0:
        return CountEvidence(None, "zero_regions_unconfirmed", "Grounding DINO", 0)
    return CountEvidence(total, "accepted_regions", "Grounding DINO accepted regions", total)


def _compare_count_evidence(
    first: CountEvidence,
    second: CountEvidence,
    relation: Optional[str],
) -> Tuple[Optional[str], bool, List[str], str]:
    unavailable = [name for name, item in (("entity_a", first), ("entity_b", second)) if item.count is None]
    comparable = (
        not unavailable
        and first.evidence_source is not None
        and first.evidence_source == second.evidence_source
    )
    if unavailable:
        return None, False, unavailable, "comparison_operand_unavailable"
    if not comparable:
        return None, False, [], "incomparable_evidence"
    operation = {
        "less": first.count < second.count,
        "more": first.count > second.count,
        "equal": first.count == second.count,
    }.get(relation)
    if operation is None:
        return None, True, [], "comparison_logic_unavailable"
    return "yes" if operation else "no", True, [], "comparison_logic_applied"


def _sve_presence_signal(response: AgentResponse, target: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    entity = get_remote_sensing_entity(target)
    sve = response.sve_result
    if entity is None or not entity.presence_uses_semantics or sve is None or not sve.available:
        return None, {"available": False, "matched_labels": []}
    labels = [prior.label for prior in sve.scene_priors]
    ranks = [labels.index(label) for label in entity.scene_prior_labels if label in labels]
    matched = [label for label in entity.scene_prior_labels if label in labels]
    if ranks and min(ranks) <= 2:
        return "yes", {"available": True, "matched_labels": matched, "best_rank": min(ranks) + 1}
    if labels and not ranks:
        return "no", {"available": True, "matched_labels": [], "top_labels_checked": len(labels)}
    return None, {"available": bool(labels), "matched_labels": matched, "best_rank": min(ranks) + 1 if ranks else None}


def _caption_presence_signal(caption: str, target: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    entity = get_remote_sensing_entity(target)
    normalized = normalize_entity_text(caption)
    if entity is None or not entity.presence_uses_semantics or not normalized:
        return None, {"available": False, "matched_terms": []}
    aliases = sorted({normalize_entity_text(alias) for alias in entity.aliases}, key=len, reverse=True)
    matched = [alias for alias in aliases if re.search(rf"\b{re.escape(alias)}\b", normalized)]
    negative = any(
        re.search(rf"\b(?:no|without|lacks?|absent)\s+(?:visible\s+)?{re.escape(alias)}\b", normalized)
        or re.search(rf"\b(?:does not|doesn t|not)\s+(?:contain|show|include)\s+(?:any\s+)?{re.escape(alias)}\b", normalized)
        for alias in aliases
    )
    if negative:
        return "no", {"available": True, "matched_terms": matched, "explicit_absence": True}
    if matched:
        return "yes", {"available": True, "matched_terms": matched[:3], "explicit_absence": False}
    return None, {"available": True, "matched_terms": []}


def _presence_decision(
    intent: QuestionIntent,
    question_response: AgentResponse,
    caption_response: AgentResponse,
    grounding_response: Optional[AgentResponse],
) -> Tuple[Optional[str], str, Optional[str], Dict[str, Any], AgentResponse]:
    entity = get_remote_sensing_entity(intent.target)
    if entity is None:
        return None, "unsupported_target", None, {"evidence_sources": {}}, question_response

    grounding_total = _grounding_region_total(grounding_response)
    signals: Dict[str, Optional[str]] = {
        "grounding": "yes" if grounding_total is not None and grounding_total > 0 else None,
    }
    sve_signal, sve_details = _sve_presence_signal(question_response, intent.target)
    caption_signal, caption_details = _caption_presence_signal(caption_response.answer or "", intent.target)
    deterministic_signal = normalize_benchmark_answer(question_response.answer, "yes_no")
    signals.update({"scene_prior": sve_signal, "caption": caption_signal, "deterministic": deterministic_signal})
    details = {
        "evidence_sources": signals,
        "grounding_region_total": grounding_total,
        "scene_prior": sve_details,
        "caption": caption_details,
    }
    if signals["grounding"] == "yes":
        return "yes", "positive_grounding", "Grounding DINO accepted regions", details, grounding_response or question_response
    positives = [name for name, signal in signals.items() if signal == "yes"]
    negatives = [name for name, signal in signals.items() if signal == "no"]
    if positives and negatives:
        return None, "conflicting_evidence", None, details, question_response
    if positives:
        preferred = next(name for name in ("scene_prior", "caption", "deterministic") if name in positives)
        reason = {"scene_prior": "positive_scene_prior", "caption": "positive_caption", "deterministic": "positive_scene_prior"}[preferred]
        source = {"scene_prior": "SatQuery Vision Encoder scene priors", "caption": "caption semantic evidence", "deterministic": "existing deterministic scene evidence"}[preferred]
        return "yes", reason, source, details, question_response
    if len(negatives) >= 2:
        return "no", "multi_source_absence", "consistent multi-source absence evidence", details, question_response
    return None, "insufficient_evidence", None, details, question_response


def _rural_urban_evidence_fusion(
    question_response: AgentResponse,
    caption_response: AgentResponse,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Fuse bounded source scores; returned scores are not probabilities."""
    urban_labels = {"urban or built-up area", "industrial area", "residential area", "road network"}
    rural_labels = {"agricultural land", "arable land", "pasture", "forest", "grassland", "vegetation"}
    source_scores: Dict[str, Dict[str, float]] = {}
    evidence_terms: Dict[str, List[str]] = {"urban": [], "rural": []}

    sve = question_response.sve_result
    if sve is not None and sve.available and sve.scene_priors:
        labels = [prior.label for prior in sve.scene_priors]
        urban_matches = [(label, 1.0 / (index + 1)) for index, label in enumerate(labels) if label in urban_labels]
        rural_matches = [(label, 1.0 / (index + 1)) for index, label in enumerate(labels) if label in rural_labels]
        source_scores["scene_prior"] = {
            "urban": max((score for _, score in urban_matches), default=0.0),
            "rural": max((score for _, score in rural_matches), default=0.0),
        }
        evidence_terms["urban"].extend(label for label, _ in urban_matches)
        evidence_terms["rural"].extend(label for label, _ in rural_matches)

    caption = normalize_entity_text(caption_response.answer or "")
    if caption:
        caption_urban = [term for term in ("urban", "built up", "residential", "industrial", "dense buildings", "dense road") if term in caption]
        caption_rural = [term for term in ("rural", "agricultural", "arable", "pasture", "forest", "grassland", "farmland", "sparse development") if term in caption]
        if caption_urban or caption_rural:
            source_scores["caption"] = {"urban": min(1.0, len(caption_urban) / 2.0), "rural": min(1.0, len(caption_rural) / 2.0)}
            evidence_terms["urban"].extend(caption_urban)
            evidence_terms["rural"].extend(caption_rural)

    statistics = question_response.vqa_details.statistics_used if question_response.vqa_details else {}
    built = statistics.get("built_up_support_percent")
    rural_support = statistics.get("rural_support_percent")
    if isinstance(built, (int, float)) and isinstance(rural_support, (int, float)):
        source_scores["scene_evidence"] = {
            "urban": max(0.0, min(1.0, float(built) / 100.0)),
            "rural": max(0.0, min(1.0, float(rural_support) / 100.0)),
        }

    weights = {"scene_prior": 0.55, "caption": 0.20, "scene_evidence": 0.25}
    available_weight = sum(weights[source] for source in source_scores)
    if available_weight <= 0:
        return None, "insufficient_evidence", {"urban_score": 0.0, "rural_score": 0.0, "margin": 0.0, "evidence_terms": evidence_terms, "source_scores": {}}
    urban_score = sum(weights[source] * scores["urban"] for source, scores in source_scores.items()) / available_weight
    rural_score = sum(weights[source] * scores["rural"] for source, scores in source_scores.items()) / available_weight
    signed_margin = urban_score - rural_score
    margin = abs(signed_margin)
    details = {
        "urban_score": round(urban_score, 6),
        "rural_score": round(rural_score, 6),
        "margin": round(margin, 6),
        "minimum_margin": 0.12,
        "scores_are_calibrated_probabilities": False,
        "source_weights": {source: weights[source] for source in source_scores},
        "source_scores": source_scores,
        "evidence_terms": evidence_terms,
    }
    if margin < 0.12:
        return None, "insufficient_evidence", details
    return ("urban" if signed_margin > 0 else "rural"), "weighted_evidence_margin", details


def _benchmark_vqa_result(
    intent: QuestionIntent,
    question_response: AgentResponse,
    caption_response: AgentResponse,
    grounding_responses: Dict[str, AgentResponse],
) -> BenchmarkVQADecision:
    """Fuse existing specialist outputs into a normalized benchmark answer.

    Returns ``answer, confidence_source, limitation, answer_source``.  No model
    is invoked here; this is a controlled adapter over the authoritative agent
    responses produced above.
    """
    category = intent.category
    if category == QuestionCategory.RURAL_URBAN_CLASSIFICATION:
        answer, reason, details = _rural_urban_evidence_fusion(question_response, caption_response)
        answer = normalize_benchmark_answer(answer, "rural_urban")
        return BenchmarkVQADecision(answer, question_response, None, "bounded rural/urban evidence fusion" if answer else None, reason, details, "vqa_rural_urban_evidence_fusion")
    if category == QuestionCategory.PRESENCE_VQA:
        answer, reason, source, details, response = _presence_decision(
            intent, question_response, caption_response, grounding_responses.get("target")
        )
        return BenchmarkVQADecision(normalize_benchmark_answer(answer, "yes_no"), response, None, source, reason, details, "vqa_presence_evidence_fusion")
    if category == QuestionCategory.COUNT_VQA:
        evidence_response = grounding_responses.get("target")
        count = _count_evidence(intent.target, evidence_response, spatial_relation=intent.spatial_relation)
        answer = normalize_benchmark_answer(str(count.count) if count.count is not None else None, "integer")
        return BenchmarkVQADecision(answer, evidence_response or question_response, _COUNT_LIMITATION, count.evidence_source, count.reason_code, {
            "canonical_target": intent.target,
            "accepted_region_count": count.accepted_region_total,
            "spatial_relation": intent.spatial_relation,
        }, "vqa_count_evidence_validation")
    if category == QuestionCategory.COMPARISON_VQA:
        first_response = grounding_responses.get("target")
        second_response = grounding_responses.get("secondary_target")
        first = _count_evidence(intent.target, first_response)
        second = _count_evidence(intent.secondary_target, second_response)
        raw_answer, comparable, unavailable, comparison_reason = _compare_count_evidence(
            first, second, intent.comparison_relation
        )
        details = {
            "entity_a": intent.target,
            "count_a": first.count,
            "entity_b": intent.secondary_target,
            "count_b": second.count,
            "relation": intent.comparison_relation,
            "comparable_evidence": comparable,
            "unavailable_operands": unavailable,
            "operand_reason_codes": {"entity_a": first.reason_code, "entity_b": second.reason_code},
        }
        response = first_response or second_response or question_response
        answer = normalize_benchmark_answer(raw_answer, "yes_no")
        return BenchmarkVQADecision(
            answer,
            response,
            _COUNT_LIMITATION,
            "Grounding DINO accepted-region evidence" if answer is not None else None,
            comparison_reason,
            details,
            "vqa_comparison_operand_resolution",
        )
    return BenchmarkVQADecision(question_response.answer, question_response, None, None, "legacy_vqa", {}, "vqa_answer_normalization")


def _numeric_confidence(response: AgentResponse) -> float:
    if response.confidence.score is not None:
        return max(0.0, min(1.0, float(response.confidence.score)))
    return {
        "high": 0.75,
        "moderate": 0.50,
        "low": 0.25,
        "unavailable": 0.0,
    }.get(response.confidence.level.value, 0.0)


def _trace(response: AgentResponse, pipeline_stage: str) -> List[Dict[str, Any]]:
    return [
        {**step.model_dump(mode="json"), "pipeline_stage": pipeline_stage}
        for step in response.execution.steps
    ]


def _selected_models(*responses: Optional[AgentResponse]) -> str:
    selected: List[str] = []
    for response in responses:
        if response is None:
            continue
        if response.sve_result is not None and response.sve_result.available:
            selected.append("satquery_vision_encoder_v1")
        selected.extend(tool for tool in response.execution.selected_tools if tool != "input_validator")
    return " + ".join(dict.fromkeys(selected)) or "satquery-agent"


def _vqa_compatibility_report(
    caption: str,
    answer: Optional[str],
    confidence_level: str,
    warnings: List[str],
) -> Dict[str, Any]:
    """Retain the legacy report shape without invoking an external provider."""
    legacy_confidence = {
        "high": "high",
        "moderate": "medium",
        "low": "low",
        "unavailable": "low",
    }.get(confidence_level, "low")
    return {
        "executive_summary": caption,
        "terrain": [],
        "structural_and_human_features": [],
        "vegetation_and_water": [],
        "image_quality": [],
        "possible_artifacts": [],
        "notes": [answer] if answer else [],
        "limitations": warnings,
        "recommended_actions": ["Review the returned evidence, execution trace, and specialist limitations."],
        "confidence": legacy_confidence,
        "disclaimer": "SatQuery outputs are supporting model and heuristic evidence, not ground truth.",
    }


async def _direct_rsvqa_endpoint_result(
    image: Image.Image,
    image_bytes: bytes,
    question: str,
    request_id: str,
    endpoint_started: float,
) -> Optional[Dict[str, Any]]:
    """Use the trained specialist as the primary RSVQA answer source when available."""
    intent = get_vqa().classify_question(question)
    if intent.category not in _BENCHMARK_VQA_CATEGORIES:
        return None
    try:
        prediction = await run_in_threadpool(
            get_rsvqa_specialist().predict,
            image,
            question,
            content_hash=hashlib.sha256(image_bytes).hexdigest(),
        )
    except RSVQASpecialistError as error:
        LOGGER.warning(
            "RSVQA Specialist v1 inference failed; using heuristic fallback: request_id=%s error=%s",
            request_id,
            error,
        )
        return None

    answer = str(prediction["answer"])
    confidence = max(0.0, min(1.0, float(prediction["confidence"])))
    task = str(prediction["task"])
    level = "high" if confidence >= 0.75 else "moderate" if confidence >= 0.50 else "low"
    limitations = [
        "Softmax confidence is not calibrated.",
        "Count outputs are learned RSVQA labels rather than physical object inventories.",
        "Captioning, SVE scene evidence, and Grounding DINO are optional supporting evidence and were not required for this prediction.",
    ]
    return {
        "report": _vqa_compatibility_report(
            "RSVQA Specialist v1 produced the primary visual-question answer.",
            answer,
            level,
            limitations,
        ),
        "provider": "satquery",
        "model": RSVQA_MODEL_USED,
        "cached": False,
        "request_id": request_id,
        "answer": answer,
        "confidence": confidence,
        "confidence_details": {
            "level": level,
            "score": confidence,
            "reason": "Maximum exported task-head softmax score; not calibrated.",
        },
        "caption": None,
        "evidence": [],
        "execution_trace": [
            {
                "tool": "query_routing",
                "status": "success",
                "duration_ms": 0,
                "parameters": {"primary_prediction_source": "rsvqa_vqa_specialist"},
                "pipeline_stage": "routing",
            },
            {
                "tool": "sve_shared_encoder",
                "status": "success",
                "duration_ms": int(prediction["encoder"].get("runtime_ms", 0)),
                "parameters": {
                    "model_reused": prediction["encoder"].get("model_reused"),
                    "image_cache_hit": prediction["encoder"].get("image_cache_hit"),
                    "text_cache_hit": prediction["encoder"].get("text_cache_hit"),
                    "device": prediction["encoder"].get("device"),
                },
                "pipeline_stage": "shared_encoder",
            },
            {
                "tool": "rsvqa_vqa_specialist",
                "status": "success",
                "duration_ms": int(prediction.get("runtime_ms", 0)),
                "parameters": {"task_head": task, "confidence_calibrated": False},
                "pipeline_stage": "primary_prediction",
            },
        ],
        "task": task,
        "model_used": RSVQA_MODEL_USED,
        "processing_time_ms": max(0, round((time.perf_counter() - endpoint_started) * 1000)),
        "question": question,
        "routed_question": question.strip(),
        "status": "success",
        "result_status": "COMPLETED",
        "warnings": limitations,
        "vqa_details": {
            "question_category": intent.category.value,
            "answer_source": RSVQA_MODEL_USED,
            "task": task,
            "logits": prediction["logits"],
            "probabilities": prediction["probabilities"],
            "confidence_calibrated": False,
        },
        "grounding_result": None,
        "sve_result": None,
    }


@app.post("/api/analysis/image", response_model=ImageAnalysisEndpointResponse)
async def analyze_image(
    image: UploadFile = File(...),
    analysis_type: str = Form(...),
    model_name: str = Form(...),
    checkpoint_name: Optional[str] = Form(None),
    display_mode: Optional[str] = Form(None),
    metrics_json: Optional[str] = Form(None),
    metadata_json: Optional[str] = Form(None),
    # Legacy field retained for existing clients during migration.
    metadata: Optional[str] = Form(None),
    question: Optional[str] = Form(
        None,
        min_length=1,
        max_length=2000,
        description="Optional RSVQA question. Omit it to preserve the legacy qualitative image-review response.",
    ),
) -> JSONResponse:
    """Qualitatively review an image and optionally run benchmarkable SatQuery VQA."""
    endpoint_started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    try:
        uploaded_mime = (image.content_type or "").lower()
        if uploaded_mime not in {"image/png", "image/jpeg", "image/webp"}:
            raise VisionAnalysisError("The generated image could not be prepared for analysis.", "INVALID_IMAGE")
        image_bytes = read_upload(image)
        if not image_bytes:
            raise VisionAnalysisError("The generated image is empty.", "INVALID_IMAGE")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise VisionAnalysisError("The generated image exceeds the AI Analysis upload limit.", "IMAGE_TOO_LARGE")
        if question is None and model_name.strip() != (VISION_ANALYSIS.settings.model or "").strip():
            raise VisionAnalysisError("The selected Gemini model is unavailable. Choose a supported model in Settings.", "UNSUPPORTED_MODEL")
        metadata_object: Dict[str, Any] = {
            "model_name": model_name.strip(), "checkpoint_name": checkpoint_name,
            "display_mode": display_mode,
        }
        for raw_value, label in ((metrics_json, "metrics_json"), (metadata_json or metadata, "metadata_json")):
            if not raw_value:
                continue
            parsed = json.loads(raw_value)
            if not isinstance(parsed, dict):
                raise ValueError("{} must be a JSON object.".format(label))
            metadata_object[label] = parsed
        LOGGER.info(
            "AI analysis request: request_id=%s endpoint=/api/analysis/image type=%s model=%s bytes=%d mime=%s checkpoint=%s display_mode=%s",
            request_id, analysis_type, model_name, len(image_bytes), uploaded_mime,
            checkpoint_name or "none", display_mode or "none",
        )
        decoded = decode_rgb(image_bytes)
        buffer = io.BytesIO()
        decoded.save(buffer, format="PNG")
        normalized_png = buffer.getvalue()
        if question is None:
            # This is the original endpoint behavior. Do not route through SatQuery or
            # alter the provider response when the optional question is absent.
            result = VISION_ANALYSIS.analyze_image(normalized_png, "image/png", analysis_type, metadata_object)
            result["request_id"] = request_id
        else:
            direct_result = await _direct_rsvqa_endpoint_result(
                decoded,
                normalized_png,
                question,
                request_id,
                endpoint_started,
            )
            if direct_result is not None:
                LOGGER.info(
                    "AI analysis completed: request_id=%s provider=satquery model=%s cached=false vqa=true",
                    request_id,
                    RSVQA_MODEL_USED,
                )
                return JSONResponse(direct_result)
            caption_response, question_response, grounding_responses, routed_question, intent = await _run_satquery_vqa(
                normalized_png,
                question,
            )
            decision = _benchmark_vqa_result(
                intent,
                question_response,
                caption_response,
                grounding_responses,
            )
            answer = decision.answer
            answer_response = decision.confidence_response
            benchmark_limitation = decision.limitation
            answer_source = decision.answer_source
            evidence = [item.model_dump(mode="json") for item in question_response.evidence]
            for grounding_response in grounding_responses.values():
                evidence.extend(item.model_dump(mode="json") for item in grounding_response.evidence)
            trace = _trace(caption_response, "captioning") + _trace(question_response, "vqa")
            for grounding_response in grounding_responses.values():
                trace.extend(_trace(grounding_response, "grounding_support"))
            trace.append({
                "tool": "vqa_evidence_fusion",
                "status": "success",
                "duration_ms": 0,
                "parameters": {
                    "evidence_items": len(evidence),
                    "grounding_used": bool(grounding_responses),
                    "benchmark_category": intent.category.value,
                },
                "pipeline_stage": "evidence_fusion",
            })
            if intent.category in _BENCHMARK_VQA_CATEGORIES:
                trace.extend([
                    {
                        "tool": "vqa_target_resolution",
                        "status": "success",
                        "duration_ms": 0,
                        "parameters": {
                            "canonical_target": intent.target,
                            "canonical_secondary_target": intent.secondary_target,
                        },
                        "pipeline_stage": "vqa_target_resolution",
                    },
                    {
                        "tool": decision.trace_stage,
                        "status": "success" if answer is not None else "partial",
                        "duration_ms": 0,
                        "parameters": {
                            "evidence_sources": sorted({
                                source for source in (
                                    decision.answer_source,
                                    *(decision.details.get("evidence_sources", {}).keys()),
                                ) if source
                            }),
                            "reason_code": decision.reason_code,
                            "comparable_evidence": decision.details.get("comparable_evidence"),
                        },
                        "pipeline_stage": decision.trace_stage,
                    },
                    {
                        "tool": "vqa_answer_normalization",
                        "status": "success" if answer is not None else "partial",
                        "duration_ms": 0,
                        "parameters": {
                            "normalized_answer": answer,
                            "reason_code": decision.reason_code,
                        },
                        "pipeline_stage": "vqa_answer_normalization",
                    },
                ])
            satquery_models = _selected_models(caption_response, question_response, *grounding_responses.values())
            caption = caption_response.answer or "The local remote-sensing captioner did not produce a caption for this image."
            warnings = list(dict.fromkeys(
                caption_response.warnings
                + question_response.warnings
                + [warning for response in grounding_responses.values() for warning in response.warnings]
            ))
            insufficient = intent.category in _BENCHMARK_VQA_CATEGORIES and answer is None
            if insufficient:
                warnings.append(
                    "Evidence was insufficient for a normalized benchmark answer; no answer was fabricated."
                )
            warnings = list(dict.fromkeys(warnings))
            confidence_details = answer_response.confidence.model_dump(mode="json")
            if benchmark_limitation:
                confidence_details["limitation"] = benchmark_limitation
            if insufficient:
                confidence_details["reason"] = (
                    f"{confidence_details.get('reason', 'No usable specialist evidence was returned.')} "
                    "The endpoint therefore returned null rather than guessing."
                )
            vqa_details = question_response.vqa_details.model_dump(mode="json") if question_response.vqa_details else None
            if vqa_details is not None and intent.category in _BENCHMARK_VQA_CATEGORIES:
                vqa_details.update({
                    "benchmark_answer_source": answer_source,
                    "benchmark_answer_normalization": "yes_no" if intent.category in {QuestionCategory.PRESENCE_VQA, QuestionCategory.COMPARISON_VQA} else "rural_urban" if intent.category == QuestionCategory.RURAL_URBAN_CLASSIFICATION else "integer",
                    "benchmark_answer": answer,
                    "reason_code": decision.reason_code,
                    "evidence_fusion": decision.details,
                })
                if intent.category == QuestionCategory.COMPARISON_VQA:
                    vqa_details["comparison"] = decision.details
                if benchmark_limitation:
                    vqa_details["counting_limitation"] = benchmark_limitation
            endpoint_task = intent.category.value if intent.category in _BENCHMARK_VQA_CATEGORIES else question_response.task.value
            result = {
                "report": _vqa_compatibility_report(caption, answer, answer_response.confidence.level.value, warnings),
                "provider": "satquery",
                "model": satquery_models,
                "cached": False,
                "request_id": request_id,
            }
            result.update(
                {
                    "answer": answer,
                    "confidence": _numeric_confidence(answer_response),
                    "confidence_details": confidence_details,
                    "caption": caption,
                    "evidence": evidence,
                    "execution_trace": trace,
                    "task": endpoint_task,
                    "model_used": satquery_models,
                    "processing_time_ms": max(0, round((time.perf_counter() - endpoint_started) * 1000)),
                    "question": question,
                    "routed_question": routed_question,
                    "status": "insufficient_evidence" if insufficient else answer_response.status.value,
                    "result_status": "INSUFFICIENT_EVIDENCE" if insufficient else answer_response.result_status,
                    "warnings": warnings,
                    "vqa_details": vqa_details,
                    "grounding_result": next(
                        (response.grounding_result.model_dump(mode="json") for response in grounding_responses.values() if response.grounding_result),
                        None,
                    ),
                    "sve_result": question_response.sve_result.model_dump(mode="json") if question_response.sve_result else None,
                }
            )
        LOGGER.info(
            "AI analysis completed: request_id=%s provider=%s model=%s cached=%s vqa=%s",
            request_id, result["provider"], result["model"], result["cached"], question is not None,
        )
        return JSONResponse(result)
    except json.JSONDecodeError as error:
        LOGGER.exception("AI analysis metadata parsing failed: request_id=%s", request_id)
        raise HTTPException(status_code=400, detail={"code": "INVALID_IMAGE", "message": "Analysis metadata must be valid JSON."}) from error
    except ValueError as error:
        LOGGER.exception("AI analysis input validation failed: request_id=%s", request_id)
        raise HTTPException(status_code=400, detail={"code": "INVALID_IMAGE", "message": str(error)}) from error
    except VisionAnalysisError as error:
        LOGGER.exception("AI analysis provider failed: request_id=%s code=%s", request_id, error.code)
        status_code = 503 if error.transient else 400
        raise HTTPException(status_code=status_code, detail={"code": error.code, "message": str(error), "request_id": request_id}) from error


@app.get("/api/settings/provider")
@app.get("/api/settings/ai-provider")
def get_provider_settings() -> JSONResponse:
    """Return non-sensitive provider status for the Settings UI."""
    return JSONResponse(PROVIDER_SETTINGS.public_status())


@app.get("/api/settings/ai-provider/models")
def get_provider_models(provider: str = "gemini") -> JSONResponse:
    try:
        return JSONResponse(PROVIDER_SETTINGS.supported_models(provider))
    except ProviderSettingsError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _test_saved_provider() -> Dict[str, Any]:
    result = VISION_ANALYSIS.test_connection()
    if result.get("success"):
        PROVIDER_SETTINGS.record_connection("connected", result.get("latency_ms"))
    elif result.get("status") == "invalid":
        PROVIDER_SETTINGS.record_connection("invalid")
    elif result.get("status") == "temporarily_unavailable":
        PROVIDER_SETTINGS.record_connection("temporarily_unavailable")
    sync_vision_settings()
    return result


@app.post("/api/settings/provider")
@app.post("/api/settings/ai-provider")
def save_provider_settings(configuration: ProviderConfigurationRequest) -> JSONResponse:
    """Verify first, then persist only a masked, server-side Gemini configuration."""
    try:
        provider = configuration.provider.strip().lower()
        LOGGER.info("Gemini configuration request: provider=%s model=%s key_length=%d endpoint=/api/settings/ai-provider", provider, (configuration.model or "").strip(), len(configuration.api_key.strip()))
        if provider != "gemini":
            raise ProviderSettingsError("Google Gemini is the currently supported provider in this configuration flow.")
        if not configuration.api_key.strip() or "\n" in configuration.api_key or "\r" in configuration.api_key:
            raise ProviderSettingsError("Enter a complete Gemini API key on one line.")
        if not (configuration.model or "").strip():
            raise ProviderSettingsError("Select a Gemini vision model.")
        supported_model_ids = {item["id"] for item in PROVIDER_SETTINGS.supported_models("gemini")["models"]}
        if configuration.model.strip() not in supported_model_ids:
            raise ProviderSettingsError("Choose a Gemini model supported by this application.")
        if provider == "gemini" and not configuration.privacy_acknowledged:
            raise ProviderSettingsError("Confirm the privacy acknowledgement before connecting Gemini.")
        candidate = ImageAnalysisService(VisionSettings(provider=provider, api_key=configuration.api_key.strip(), model=(configuration.model or "").strip()))
        checked = candidate.test_connection()
        if not checked.get("success"):
            raise ProviderSettingsError(str(checked.get("message") or "Connection test failed."))
        status = PROVIDER_SETTINGS.save(provider, configuration.api_key, configuration.model)
        sync_vision_settings()
        return JSONResponse({"success": True, "message": "Google Gemini connected successfully." if provider == "gemini" else "Configuration saved successfully.", **status, "latency_ms": checked.get("latency_ms")})
    except ProviderSettingsError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/settings/provider")
@app.delete("/api/settings/ai-provider")
def delete_provider_settings() -> JSONResponse:
    try:
        status = PROVIDER_SETTINGS.delete()
        sync_vision_settings()
        return JSONResponse({"success": True, "message": "Gemini configuration removed.", **status})
    except ProviderSettingsError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/settings/test")
@app.post("/api/settings/ai-provider/test")
def test_provider_settings() -> JSONResponse:
    return JSONResponse(_test_saved_provider())
