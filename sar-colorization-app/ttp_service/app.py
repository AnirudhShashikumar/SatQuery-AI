"""FastAPI boundary for persistent TTP CUDA inference."""

from __future__ import annotations

import asyncio
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from skimage.measure import label, regionprops

from .artifacts import ARTIFACT_STORE
from .lifecycle import MODEL_LIFECYCLE, LifecycleError, ModelLifecycle
from .schemas import HealthResponse, PredictionResponse, RuntimeMetrics
from .security import ValidationError, validate_image, validate_pair, validate_request_id


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_error(error: Exception) -> HTTPException:
    if isinstance(error, ValidationError):
        return HTTPException(status_code=422, detail={"code": error.code, "message": error.message})
    if isinstance(error, LifecycleError):
        status = 503 if error.code in {"MODEL_NOT_READY", "MODEL_LOAD_FAILED", "CUDA_OOM"} else 500
        return HTTPException(status_code=status, detail={"code": error.code, "message": error.message})
    return HTTPException(status_code=500, detail={"code": "PREDICTION_FAILED", "message": "TTP prediction failed safely."})


def create_app(lifecycle: ModelLifecycle = MODEL_LIFECYCLE) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if _enabled(os.getenv("TTP_LOAD_ON_STARTUP"), False):
            try:
                await asyncio.to_thread(lifecycle.load)
                if _enabled(os.getenv("TTP_WARMUP_ON_STARTUP"), False):
                    warm = io.BytesIO()
                    Image.new("RGB", (512, 512), (0, 0, 0)).save(warm, "PNG")
                    await asyncio.to_thread(lifecycle.predict, warm.getvalue(), warm.getvalue(), ".png", ".png")
            except LifecycleError:
                pass
        yield

    app = FastAPI(title="GeoVision TTP Change Detector", version="1.0.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        lifecycle_state = lifecycle.state
        status = {"ready": "ready", "loading": "loading", "unloaded": "unavailable", "failed": "failed"}[lifecycle_state]
        return HealthResponse(
            status=status,
            checkpoint_verified=lifecycle.checkpoint_verified,
            device="cuda" if lifecycle_state == "ready" else "unavailable",
            lifecycle=lifecycle_state,
            model_load_count=lifecycle.model_load_count,
            inference_count=lifecycle.inference_count,
            model_reuse_count=lifecycle.model_reuse_count,
        )

    @app.get("/artifacts/{artifact_id}")
    async def artifact(artifact_id: str) -> Response:
        item = ARTIFACT_STORE.get(artifact_id)
        if item is None:
            raise HTTPException(status_code=404, detail={"code": "ARTIFACT_UNAVAILABLE", "message": "The requested artifact is unavailable or expired."})
        return Response(item.data, media_type="image/png", headers={"Cache-Control": "private, max-age=300", "X-Content-Type-Options": "nosniff"})

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        earlier_image: UploadFile = File(...),
        later_image: UploadFile = File(...),
        request_id: Optional[str] = Form(None),
    ) -> PredictionResponse:
        started = time.perf_counter()
        earlier_rgb = later_rgb = None
        try:
            safe_request_id = validate_request_id(request_id)
            earlier = validate_image(await earlier_image.read(), earlier_image.filename, earlier_image.content_type)
            later = validate_image(await later_image.read(), later_image.filename, later_image.content_type)
            earlier_rgb, later_rgb = earlier.rgb, later.rgb
            validate_pair(earlier, later)
            mask, reused, inference_ms, gpu = await asyncio.to_thread(
                lifecycle.predict, earlier.data, later.data, earlier.extension, later.extension
            )
            if mask.shape != (earlier.height, earlier.width):
                raise LifecycleError("INVALID_MASK_DIMENSIONS", "TTP returned a mask with unexpected dimensions.")
            binary = mask.astype(np.uint8)
            components = regionprops(label(binary, connectivity=2))
            changed_pixels = int(binary.sum())
            display_mask = np.zeros((earlier.height, earlier.width, 3), dtype=np.uint8)
            display_mask[binary.astype(bool)] = (255, 72, 72)
            overlay = np.asarray(later.rgb, dtype=np.float32)
            overlay[binary.astype(bool)] = 0.45 * overlay[binary.astype(bool)] + 0.55 * np.array([255.0, 32.0, 32.0])
            raw_ref = ARTIFACT_STORE.put_image(Image.fromarray(binary * 255), "raw_binary_mask")
            display_ref = ARTIFACT_STORE.put_image(Image.fromarray(display_mask), "display_mask")
            overlay_ref = ARTIFACT_STORE.put_image(Image.fromarray(np.round(overlay).astype(np.uint8)), "overlay")
            total_ms = max(0, round((time.perf_counter() - started) * 1000))
            return PredictionResponse(
                request_id=safe_request_id,
                input_width=earlier.width,
                input_height=earlier.height,
                changed_pixels=changed_pixels,
                changed_percentage=round(changed_pixels * 100.0 / binary.size, 6),
                region_count=len(components),
                largest_region_pixels=max((int(region.area) for region in components), default=0),
                runtime=RuntimeMetrics(
                    model_load_ms=lifecycle.model_load_ms,
                    inference_ms=inference_ms,
                    total_request_ms=total_ms,
                    **gpu,
                ),
                reused_model=reused,
                warnings=[],
                trace=[
                    {"stage": "input_validation", "status": "success", "width": earlier.width, "height": earlier.height},
                    {"stage": "model_reuse" if reused else "model_ready", "status": "success"},
                    {"stage": "ttp_inference", "status": "success", "runtime_ms": inference_ms},
                    {
                        "stage": "probability_map_generation",
                        "status": "success",
                        "decision_threshold": 0.5,
                        "source": getattr(lifecycle, "probability_source", None),
                    },
                    {"stage": "mask_validation", "status": "success", "changed_pixels": changed_pixels},
                ],
                artifacts=[raw_ref, display_ref, overlay_ref],
            )
        except Exception as error:
            raise _safe_error(error) from error
        finally:
            if earlier_rgb is not None:
                earlier_rgb.close()
            if later_rgb is not None:
                later_rgb.close()

    return app


app = create_app()
