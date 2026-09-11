"""Defensive client for the isolated TTP CUDA service.

The adapter never exposes configuration, URLs, paths, or remote exceptions in
public GeoVision contracts. Every failure is converted to a bounded typed code
so deterministic analysis can remain available.
"""

from __future__ import annotations

import io
import math
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from skimage.measure import label, regionprops


MODEL_ID = "TTP"
CHECKPOINT_NAME = "epoch_260.pth"
CHECKPOINT_FINGERPRINT = "60294429b3d"
ARCHITECTURE = "SAM ViT-L + LoRA SiamEncoderDecoder"
TRAINING_DATASET = "LEVIR-CD"
DETERMINISTIC_VERSION = "bitemporal-change-1.0"
DOMAIN_WARNING = (
    "TTP was trained on LEVIR-CD building-change imagery. Generalization may be weaker for vegetation change, "
    "flooding, seasonal variation, unfamiliar sensors, SAR imagery or non-urban scenes."
)


def _positive_float(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def ttp_enabled() -> bool:
    return os.getenv("TTP_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def default_mode() -> str:
    value = os.getenv("TTP_DEFAULT_MODE", "hybrid").strip().lower()
    return value if value in {"deterministic", "ttp", "hybrid"} else "hybrid"


class TTPClientError(RuntimeError):
    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


@dataclass
class TTPClientResult:
    mask: np.ndarray
    changed_percentage: float
    changed_pixels: int
    region_count: int
    largest_region_pixels: int
    runtime_ms: int
    model_load_ms: int
    reused_model: bool
    device: str
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    gpu_peak_mb: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    service_trace: list[dict[str, object]] = field(default_factory=list)


class TTPServiceClient:
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()
        self._health_lock = threading.RLock()
        self._cached_health: Optional[Dict[str, Any]] = None
        self._health_expires = 0.0
        self._metrics = Counter()
        self._runtime_total_ms = 0
        self._last_error: Optional[str] = None
        self._last_error_code: Optional[str] = None
        self._last_error_expires = 0.0

    def _cache_health_error(self, error: TTPClientError) -> None:
        with self._health_lock:
            self._last_error = error.public_message
            self._last_error_code = error.code
            self._last_error_expires = time.monotonic() + _positive_float("TTP_HEALTH_CACHE_SECONDS", 10.0)

    @property
    def _base_url(self) -> str:
        value = os.getenv("TTP_SERVICE_URL", "http://127.0.0.1:8000").strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise TTPClientError("INVALID_CONFIGURATION", "TTP service configuration is invalid.")
        return value

    @property
    def _timeouts(self) -> tuple[float, float]:
        return (_positive_float("TTP_CONNECT_TIMEOUT_SECONDS", 2.0), _positive_float("TTP_READ_TIMEOUT_SECONDS", 45.0))

    def _read_json(self, response: requests.Response, maximum: int = 512 * 1024) -> Dict[str, Any]:
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type != "application/json":
            raise TTPClientError("MALFORMED_RESPONSE", "TTP returned an unexpected response type.")
        declared = response.headers.get("content-length")
        if declared and int(declared) > maximum:
            raise TTPClientError("RESPONSE_TOO_LARGE", "TTP returned an oversized response.")
        data = response.content
        if len(data) > maximum:
            raise TTPClientError("RESPONSE_TOO_LARGE", "TTP returned an oversized response.")
        try:
            payload = response.json()
        except ValueError as error:
            raise TTPClientError("MALFORMED_RESPONSE", "TTP returned malformed JSON.") from error
        if not isinstance(payload, dict):
            raise TTPClientError("MALFORMED_RESPONSE", "TTP returned an invalid response envelope.")
        return payload

    def health(self, force: bool = False) -> Dict[str, Any]:
        if not ttp_enabled():
            raise TTPClientError("DISABLED", "TTP is disabled; deterministic analysis remains active.")
        now = time.monotonic()
        with self._health_lock:
            if not force and self._cached_health is not None and now < self._health_expires:
                return dict(self._cached_health)
            if not force and self._last_error and self._last_error_code and now < self._last_error_expires:
                raise TTPClientError(self._last_error_code, self._last_error)
        try:
            response = self._session.get(urljoin(self._base_url + "/", "health"), timeout=self._timeouts)
        except requests.Timeout as error:
            self._metrics["timeouts"] += 1
            failure = TTPClientError("TIMEOUT", "TTP health check timed out; deterministic fallback was used.")
            self._cache_health_error(failure)
            raise failure from error
        except requests.RequestException as error:
            self._metrics["failures"] += 1
            failure = TTPClientError("UNAVAILABLE", "TTP is unavailable; deterministic fallback was used.")
            self._cache_health_error(failure)
            raise failure from error
        if response.status_code != 200:
            failure = TTPClientError("UNAVAILABLE", "TTP is unavailable; deterministic fallback was used.")
            self._cache_health_error(failure)
            raise failure
        payload = self._read_json(response)
        required = {
            "status": "ready", "service": "ttp_change_detector", "model": MODEL_ID,
            "training_dataset": TRAINING_DATASET, "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT,
            "checkpoint_verified": True, "device": "cuda", "lifecycle": "ready",
        }
        if any(payload.get(key) != value for key, value in required.items()):
            failure = TTPClientError("UNHEALTHY", "TTP health or provenance validation failed; deterministic fallback was used.")
            self._cache_health_error(failure)
            raise failure
        with self._health_lock:
            self._cached_health = dict(payload)
            self._health_expires = now + _positive_float("TTP_HEALTH_CACHE_SECONDS", 10.0)
            self._last_error = None
            self._last_error_code = None
            self._last_error_expires = 0.0
        return payload

    def checkpoint(self) -> str:
        return CHECKPOINT_NAME

    def device(self) -> Optional[str]:
        with self._health_lock:
            if self._cached_health and self._cached_health.get("lifecycle") == "ready":
                return str(self._cached_health.get("device") or "cuda")
        return None

    def health_payload(self, force: bool = False) -> Dict[str, Any]:
        """Return complete, path-free lifecycle state for the main agent health API."""
        if not ttp_enabled():
            return {
                "status": "disabled", "lifecycle": "disabled", "device": None,
                "checkpoint": CHECKPOINT_NAME, "loaded": False, "load_count": 0,
                "reuse_count": 0, "inference_count": 0, "errors": [],
            }
        try:
            remote = self.health(force=force)
            lifecycle = str(remote.get("lifecycle", "unavailable"))
            return {
                "status": str(remote.get("status", lifecycle)),
                "lifecycle": lifecycle,
                "device": str(remote.get("device")) if remote.get("device") else None,
                "checkpoint": CHECKPOINT_NAME,
                "loaded": lifecycle == "ready",
                "load_count": max(0, int(remote.get("model_load_count", 0))),
                "reuse_count": max(0, int(remote.get("model_reuse_count", 0))),
                "inference_count": max(0, int(remote.get("inference_count", 0))),
                "errors": [],
            }
        except TTPClientError as error:
            self._cache_health_error(error)
            return {
                "status": "failed" if error.code in {"UNHEALTHY", "MODEL_LOAD_FAILED"} else "unavailable",
                "lifecycle": "failed" if error.code in {"UNHEALTHY", "MODEL_LOAD_FAILED"} else "unavailable",
                "device": None,
                "checkpoint": CHECKPOINT_NAME,
                "loaded": False,
                "load_count": 0,
                "reuse_count": int(self._metrics["reuses"]),
                "inference_count": int(self._metrics["inferences"]),
                "errors": [error.public_message],
            }

    def _fetch_mask(self, endpoint: str, width: int, height: int) -> np.ndarray:
        if not endpoint.startswith("/artifacts/") or len(endpoint.split("/")[-1]) != 32:
            raise TTPClientError("INVALID_ARTIFACT", "TTP returned an invalid artifact reference.")
        try:
            response = self._session.get(urljoin(self._base_url + "/", endpoint.lstrip("/")), timeout=self._timeouts, stream=True)
        except requests.Timeout as error:
            raise TTPClientError("TIMEOUT", "TTP artifact retrieval timed out; deterministic fallback was used.") from error
        except requests.RequestException as error:
            raise TTPClientError("INVALID_ARTIFACT", "TTP mask artifact could not be retrieved.") from error
        if response.status_code != 200 or response.headers.get("content-type", "").split(";", 1)[0].lower() != "image/png":
            raise TTPClientError("INVALID_ARTIFACT", "TTP mask artifact was unavailable or invalid.")
        maximum = width * height + 2 * 1024 * 1024
        data = response.content
        if len(data) > maximum:
            raise TTPClientError("INVALID_ARTIFACT", "TTP mask artifact exceeded the safe response limit.")
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.size != (width, height) or image.mode not in {"1", "L"}:
                    raise TTPClientError("INVALID_MASK", "TTP mask dimensions or mode were invalid.")
                values = np.asarray(image.convert("L"))
        except TTPClientError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise TTPClientError("INVALID_MASK", "TTP mask could not be decoded.") from error
        unique = set(np.unique(values).tolist())
        if not unique.issubset({0, 1, 255}):
            raise TTPClientError("INVALID_MASK", "TTP mask contained non-binary values.")
        return values > 0

    def predict(self, earlier: bytes, later: bytes, earlier_name: str, later_name: str, width: int, height: int, request_id: str) -> TTPClientResult:
        self.health()
        started = time.perf_counter()
        files = {
            "earlier_image": (earlier_name, earlier, "application/octet-stream"),
            "later_image": (later_name, later, "application/octet-stream"),
        }
        try:
            response = self._session.post(urljoin(self._base_url + "/", "predict"), files=files, data={"request_id": request_id}, timeout=self._timeouts)
        except requests.Timeout as error:
            self._metrics["timeouts"] += 1
            raise TTPClientError("TIMEOUT", "TTP inference timed out; deterministic fallback was used.") from error
        except requests.RequestException as error:
            self._metrics["failures"] += 1
            raise TTPClientError("UNAVAILABLE", "TTP inference was unavailable; deterministic fallback was used.") from error
        if response.status_code != 200:
            code = "INFERENCE_FAILED"
            try:
                detail = response.json().get("detail", {})
                if isinstance(detail, dict) and detail.get("code") == "CUDA_OOM":
                    code = "CUDA_OOM"
            except ValueError:
                pass
            if code == "CUDA_OOM":
                self._metrics["oom"] += 1
                raise TTPClientError(code, "TTP ran out of CUDA memory; deterministic fallback was used.")
            self._metrics["failures"] += 1
            raise TTPClientError(code, "TTP inference failed safely; deterministic fallback was used.")
        payload = self._read_json(response)
        required_values = {
            "status": "success", "model": MODEL_ID, "training_dataset": TRAINING_DATASET,
            "checkpoint": CHECKPOINT_NAME, "checkpoint_fingerprint": CHECKPOINT_FINGERPRINT, "device": "cuda",
            "input_width": width, "input_height": height,
        }
        if any(payload.get(key) != value for key, value in required_values.items()):
            raise TTPClientError("MALFORMED_RESPONSE", "TTP prediction provenance or dimensions were invalid.")
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise TTPClientError("MALFORMED_RESPONSE", "TTP returned no artifact list.")
        raw = next((item for item in artifacts if isinstance(item, dict) and item.get("kind") == "raw_binary_mask"), None)
        if raw is None or not isinstance(raw.get("endpoint"), str):
            raise TTPClientError("INVALID_ARTIFACT", "TTP returned no raw binary mask artifact.")
        mask = self._fetch_mask(raw["endpoint"], width, height)
        runtime = payload.get("runtime") or {}
        numeric = [payload.get("changed_percentage"), payload.get("changed_pixels"), payload.get("region_count"), payload.get("largest_region_pixels"), runtime.get("inference_ms")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in numeric):
            raise TTPClientError("MALFORMED_RESPONSE", "TTP returned invalid statistics.")
        memory_values = [runtime.get(name) for name in ("gpu_allocated_mb", "gpu_reserved_mb", "gpu_peak_mb")]
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0) for value in memory_values):
            raise TTPClientError("MALFORMED_RESPONSE", "TTP returned invalid memory metrics.")
        changed_pixels = int(mask.sum())
        if int(payload["changed_pixels"]) != changed_pixels:
            raise TTPClientError("INVALID_MASK", "TTP mask and reported statistics were inconsistent.")
        expected_percentage = changed_pixels * 100.0 / mask.size
        components = regionprops(label(mask, connectivity=2))
        largest = max((int(region.area) for region in components), default=0)
        if abs(float(payload["changed_percentage"]) - expected_percentage) > 1e-4 or int(payload["region_count"]) != len(components) or int(payload["largest_region_pixels"]) != largest:
            raise TTPClientError("INVALID_MASK", "TTP mask and reported region statistics were inconsistent.")
        runtime_ms = int(runtime.get("inference_ms", 0))
        self._metrics["inferences"] += 1
        if payload.get("reused_model"):
            self._metrics["reuses"] += 1
        self._runtime_total_ms += runtime_ms
        return TTPClientResult(
            mask=mask,
            changed_percentage=float(payload["changed_percentage"]),
            changed_pixels=changed_pixels,
            region_count=int(payload["region_count"]),
            largest_region_pixels=int(payload["largest_region_pixels"]),
            runtime_ms=runtime_ms,
            model_load_ms=int(runtime.get("model_load_ms", 0)),
            reused_model=bool(payload.get("reused_model")),
            device="cuda",
            gpu_allocated_mb=float(runtime["gpu_allocated_mb"]) if runtime.get("gpu_allocated_mb") is not None else None,
            gpu_reserved_mb=float(runtime["gpu_reserved_mb"]) if runtime.get("gpu_reserved_mb") is not None else None,
            gpu_peak_mb=float(runtime["gpu_peak_mb"]) if runtime.get("gpu_peak_mb") is not None else None,
            warnings=[str(item) for item in payload.get("warnings", []) if isinstance(item, str)],
            limitations=[str(item) for item in payload.get("limitations", []) if isinstance(item, str)],
            service_trace=[item for item in payload.get("trace", []) if isinstance(item, dict)],
        )

    def metrics(self) -> Dict[str, int | float | None]:
        count = int(self._metrics["inferences"])
        return {
            "inference_count": count,
            "model_reuse_count": int(self._metrics["reuses"]),
            "failure_count": int(self._metrics["failures"]),
            "timeout_count": int(self._metrics["timeouts"]),
            "oom_count": int(self._metrics["oom"]),
            "average_runtime_ms": round(self._runtime_total_ms / count, 1) if count else None,
        }


TTP_CLIENT = TTPServiceClient()
