"""Public, path-free API contracts for the TTP CUDA service."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


MODEL_ID = "TTP"
ARCHITECTURE = "SAM ViT-L + LoRA SiamEncoderDecoder"
TRAINING_DATASET = "LEVIR-CD"
CHECKPOINT_NAME = "epoch_260.pth"
CHECKPOINT_SHA256 = "60294429b3d22310e1451b1059b953b44323adfe3af4eae1d75ae1709e610cb9"
CHECKPOINT_FINGERPRINT = CHECKPOINT_SHA256[:12]
LIMITATIONS = [
    "Binary change detection only",
    "Optimized primarily for optical building-change imagery",
    "No semantic cause inference",
]


class ArtifactReference(BaseModel):
    id: str
    kind: Literal["raw_binary_mask", "display_mask", "overlay"]
    media_type: Literal["image/png"] = "image/png"
    endpoint: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(gt=0)


class RuntimeMetrics(BaseModel):
    model_load_ms: int = Field(ge=0)
    inference_ms: int = Field(ge=0)
    total_request_ms: int = Field(ge=0)
    gpu_allocated_mb: Optional[float] = Field(default=None, ge=0)
    gpu_reserved_mb: Optional[float] = Field(default=None, ge=0)
    gpu_peak_mb: Optional[float] = Field(default=None, ge=0)


class HealthResponse(BaseModel):
    status: Literal["ready", "loading", "unavailable", "failed"]
    service: str = "ttp_change_detector"
    model: str = MODEL_ID
    architecture: str = ARCHITECTURE
    training_dataset: str = TRAINING_DATASET
    checkpoint_verified: bool
    checkpoint_fingerprint: str = CHECKPOINT_FINGERPRINT
    device: str
    lifecycle: Literal["unloaded", "loading", "ready", "failed"]
    model_load_count: int = Field(ge=0)
    inference_count: int = Field(ge=0)
    model_reuse_count: int = Field(ge=0)
    limitations: list[str] = Field(default_factory=lambda: list(LIMITATIONS))


class PredictionResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: Optional[str] = None
    model: str = MODEL_ID
    architecture: str = ARCHITECTURE
    training_dataset: str = TRAINING_DATASET
    checkpoint: str = CHECKPOINT_NAME
    checkpoint_fingerprint: str = CHECKPOINT_FINGERPRINT
    device: Literal["cuda"] = "cuda"
    input_width: int = Field(gt=0)
    input_height: int = Field(gt=0)
    changed_pixels: int = Field(ge=0)
    changed_percentage: float = Field(ge=0, le=100)
    region_count: int = Field(ge=0)
    largest_region_pixels: int = Field(ge=0)
    runtime: RuntimeMetrics
    reused_model: bool
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(LIMITATIONS))
    trace: list[dict[str, object]] = Field(default_factory=list)
    artifacts: list[ArtifactReference]
