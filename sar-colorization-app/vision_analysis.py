"""Provider-neutral, server-side vision analysis for rendered SAR products.

This module deliberately operates on exported display images only.  It never
touches model tensors, checkpoints, preprocessing, or metric calculations.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

LOGGER = logging.getLogger("sar-colorization.vision")
DEBUG_ENABLED = os.getenv("GEOVISION_DEBUG", "").strip().lower() in {"1", "true", "yes"}


ANALYSIS_TYPES = {
    "pix2pix",
    "sarfusionformer_raw",
    "sarfusionformer_enhanced",
    "sarfusionformer_corrected",
    "ground_truth",
    "difference",
    "comparison",
}

SYSTEM_PROMPT = """You are an assistant supporting qualitative remote-sensing image review.
Analyze only visible patterns in the supplied rendered image. Do not claim that generated
SAR-to-optical imagery is real ground truth, do not infer precise geography, dates, land
ownership, activity, or identities, and do not make safety-critical decisions. Clearly
distinguish observations from plausible interpretations. Be especially cautious around
model hallucinations, synthetic colour, blur, tiling, and contrast enhancement. Return
only valid JSON matching the requested schema."""

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "terrain": {"type": "array", "items": {"type": "string"}},
        "structural_and_human_features": {"type": "array", "items": {"type": "string"}},
        "vegetation_and_water": {"type": "array", "items": {"type": "string"}},
        "image_quality": {"type": "array", "items": {"type": "string"}},
        "possible_artifacts": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "disclaimer": {"type": "string"},
    },
    "required": [
        "executive_summary", "terrain", "structural_and_human_features",
        "vegetation_and_water", "image_quality", "possible_artifacts", "notes",
        "limitations", "recommended_actions", "confidence", "disclaimer",
    ],
}


@dataclass(frozen=True)
class VisionSettings:
    provider: str
    api_key: Optional[str]
    model: str

    @classmethod
    def from_environment(cls) -> "VisionSettings":
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.isfile(env_file):
            with open(env_file, "r", encoding="utf-8") as environment:
                for line in environment:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
        return cls(
            provider=(os.getenv("VISION_PROVIDER", "").strip().lower() or ("gemini" if os.getenv("GEMINI_API_KEY") else "")),
            api_key=os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"),
            model=(os.getenv("VISION_MODEL") if os.getenv("VISION_PROVIDER", "").strip().lower() != "gemini" else os.getenv("GEMINI_MODEL")) or os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.5-flash"),
        )

    @property
    def configured(self) -> bool:
        return self.provider in {"openai", "gemini"} and bool(self.api_key) and bool(self.model)


class VisionAnalysisError(RuntimeError):
    """Expected configuration, provider, or schema error for this optional feature."""

    def __init__(self, message: str, code: str = "PROVIDER_ERROR", transient: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.transient = transient

class ImageAnalysisService:
    """Small provider interface; add providers here without exposing keys to the UI."""

    def __init__(self, settings: Optional[VisionSettings] = None) -> None:
        self.settings = settings or VisionSettings.from_environment()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.settings.configured,
            "provider": self.settings.provider or None,
            "model": self.settings.model if self.settings.configured else None,
        }

    def _gemini_client(self):
        return genai.Client(api_key=str(self.settings.api_key))

    def _debug(self, **values: Any) -> Dict[str, Any]:
        safe = {key: value for key, value in values.items() if key not in {"api_key", "image_bytes", "response_text"}}
        return safe if DEBUG_ENABLED else {}

    def _gemini_exception(self, error: Exception) -> VisionAnalysisError:
        """Turn SDK failures into safe, actionable UI error codes without secrets."""
        http_code = getattr(error, "code", None)
        details = (str(getattr(error, "details", "")) + " " + str(error)).lower()
        if http_code == 401 or "api_key_invalid" in details or "api key not valid" in details:
            return VisionAnalysisError("The saved Gemini API key is invalid or no longer authorized.", "INVALID_API_KEY")
        if http_code == 403 or "permission denied" in details or "permission" in details:
            return VisionAnalysisError("This API key does not have access to the selected Gemini model.", "MODEL_PERMISSION_DENIED")
        if http_code == 404 or "not found" in details or ("model" in details and "supported" in details):
            return VisionAnalysisError("The selected Gemini model is unavailable. Choose a supported model in Settings.", "UNSUPPORTED_MODEL")
        if http_code == 429:
            if "quota" in details or "resource_exhausted" in details:
                return VisionAnalysisError("Your Gemini quota has been reached. Try again later or use another key.", "QUOTA_EXCEEDED", True)
            return VisionAnalysisError("Gemini is temporarily rate-limited. Wait briefly and retry.", "RATE_LIMITED", True)
        if http_code in {408, 504} or "timeout" in details or "timed out" in details:
            return VisionAnalysisError("Gemini did not respond in time. Your reconstruction result is unaffected.", "TIMEOUT", True)
        if "connect" in details or "network" in details or "dns" in details or "name resolution" in details:
            return VisionAnalysisError("GeoVision could not reach Google Gemini. Check the network and retry.", "NETWORK_ERROR", True)
        if isinstance(error, genai_errors.ServerError) or (isinstance(http_code, int) and http_code >= 500):
            return VisionAnalysisError("Google Gemini is temporarily unavailable. Try again shortly.", "GOOGLE_API_UNAVAILABLE", True)
        return VisionAnalysisError("Gemini returned an unexpected provider error. The reconstruction result is unaffected.", "UNKNOWN_PROVIDER_ERROR")

    def test_connection(self) -> Dict[str, Any]:
        if not self.settings.configured:
            return {"success": False, "status": "not_configured", "message": "AI analysis requires an API key.", "error_code": "NOT_CONFIGURED"}
        started = __import__("time").perf_counter()
        try:
            if self.settings.provider == "gemini":
                LOGGER.info("Gemini connection test: model=%s key_length=%d endpoint=sdk.generate_content", self.settings.model, len(str(self.settings.api_key)))
                client = self._gemini_client()
                response = client.models.generate_content(
                    model=self.settings.model, contents="Reply with OK.",
                    config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=8),
                )
                LOGGER.info("Gemini connection test completed: model=%s response=%s", self.settings.model, bool(getattr(response, "text", None)))
            elif self.settings.provider == "openai":
                response = requests.get(
                    "https://api.openai.com/v1/models/{}".format(self.settings.model),
                    headers={"Authorization": "Bearer {}".format(self.settings.api_key)}, timeout=15
                )
                if not response.ok:
                    raise VisionAnalysisError("The saved API key could not access the selected model.", "INVALID_API_KEY")
            else:
                raise VisionAnalysisError("Unsupported AI provider.", "UNSUPPORTED_PROVIDER")
            return {"success": True, "provider": "Google Gemini" if self.settings.provider == "gemini" else "OpenAI", "model": self.settings.model, "status": "connected", "latency_ms": round((__import__("time").perf_counter() - started) * 1000), "message": "Connection successful.", "debug": self._debug(model=self.settings.model, endpoint="sdk.generate_content", status="received", latency_ms=round((time.perf_counter() - started) * 1000))}
        except (genai_errors.APIError, requests.RequestException, TimeoutError) as error:
            mapped = self._gemini_exception(error) if self.settings.provider == "gemini" else VisionAnalysisError("The provider could not be reached.", "NETWORK_FAILURE", True)
            LOGGER.exception("Gemini connection test failed: model=%s key_length=%d code=%s", self.settings.model, len(str(self.settings.api_key)), mapped.code)
            return {"success": False, "provider": self.settings.provider, "status": "temporarily_unavailable" if mapped.transient else "invalid", "error_code": mapped.code, "message": str(mapped), "debug": self._debug(model=self.settings.model, endpoint="sdk.generate_content", error_code=mapped.code)}
        except VisionAnalysisError as error:
            LOGGER.exception("Provider connection test failed: code=%s", error.code)
            return {"success": False, "provider": self.settings.provider, "status": "temporarily_unavailable" if error.transient else "invalid", "error_code": error.code, "message": str(error)}
        except Exception as error:
            mapped = self._gemini_exception(error) if self.settings.provider == "gemini" else VisionAnalysisError("The provider could not be reached.", "NETWORK_FAILURE", True)
            LOGGER.exception("Unexpected Gemini connection-test failure: model=%s key_length=%d code=%s", self.settings.model, len(str(self.settings.api_key)), mapped.code)
            return {"success": False, "provider": self.settings.provider, "status": "temporarily_unavailable" if mapped.transient else "invalid", "error_code": mapped.code, "message": str(mapped), "debug": self._debug(model=self.settings.model, endpoint="sdk.generate_content", error_code=mapped.code)}

    def analyze_image(
        self, image_bytes: bytes, mime_type: str, analysis_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if analysis_type not in ANALYSIS_TYPES:
            raise VisionAnalysisError("Unsupported analysis type: {}".format(analysis_type))
        if not self.settings.configured:
            raise VisionAnalysisError("Connect Google Gemini before using AI Analysis.", "PROVIDER_NOT_CONFIGURED")
        if self.settings.provider not in {"openai", "gemini"}:
            raise VisionAnalysisError("Unsupported AI provider.", "UNSUPPORTED_PROVIDER")

        cache_key = hashlib.sha256(
            image_bytes + self.settings.model.encode("utf-8") + analysis_type.encode("utf-8") + json.dumps(metadata or {}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key in self._cache:
            return {**self._cache[cache_key], "cached": True}
        if self.settings.provider == "gemini":
            result = self._analyze_gemini(image_bytes, mime_type, analysis_type, metadata or {})
            self._cache[cache_key] = result
            return result

        encoded = base64.b64encode(image_bytes).decode("ascii")
        context = {
            "analysis_type": analysis_type,
            "metadata": metadata or {},
            "instruction": (
                "Provide a qualitative review of this rendered image. This may be a generated "
                "SAR-to-optical prediction, an enhanced display, a colour-corrected rendering, "
                "or reference imagery. Describe only visual evidence and preserve scientific caution."
            ),
        }
        payload = {
            "model": self.settings.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(context)},
                        {"type": "input_image", "image_url": "data:{};base64,{}".format(mime_type, encoded), "detail": "high"},
                    ],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "remote_sensing_image_review",
                    "strict": True,
                    "schema": SCHEMA,
                }
            },
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer {}".format(self.settings.api_key), "Content-Type": "application/json"},
                json=payload,
                timeout=75,
            )
        except requests.RequestException as error:
            raise VisionAnalysisError("The vision provider could not be reached.") from error
        if not response.ok:
            if response.status_code == 401:
                raise VisionAnalysisError("OpenAI rejected the saved API key. Replace it in Settings → AI Providers.")
            if response.status_code == 429:
                raise VisionAnalysisError(
                    "OpenAI has no available API quota for this project. Add billing or use a funded API key, then test the connection in Settings → AI Providers."
                )
            if response.status_code == 403:
                raise VisionAnalysisError("The saved API key does not have access to the configured model. Review the project permissions in Settings → AI Providers.")
            raise VisionAnalysisError("The vision provider could not complete this request (HTTP {}). Try again or test the connection in Settings → AI Providers.".format(response.status_code))

        try:
            response_data = response.json()
            output_text = response_data.get("output_text")
            if not output_text:
                output_text = next(
                    content["text"]
                    for item in response_data.get("output", [])
                    for content in item.get("content", [])
                    if content.get("type") == "output_text" and content.get("text")
                )
            report = json.loads(output_text)
        except (KeyError, StopIteration, TypeError, ValueError) as error:
            raise VisionAnalysisError("Vision provider returned an invalid structured analysis.") from error
        result = {"report": report, "provider": self.settings.provider, "model": self.settings.model, "cached": False}
        self._cache[cache_key] = result
        return result

    def _analyze_gemini(self, image_bytes: bytes, mime_type: str, analysis_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise VisionAnalysisError("Unsupported image format. Use PNG, JPEG, or WEBP.", "INVALID_IMAGE")
        if len(image_bytes) > 20 * 1024 * 1024:
            raise VisionAnalysisError("Image too large for AI Analysis. Use an image smaller than 20 MB.", "IMAGE_TOO_LARGE")
        prompt = """You are a professional remote sensing scientist. Analyze this reconstructed optical satellite image. Assess overall scene, land cover, vegetation, water, roads, buildings, urban density, agriculture, possible flood evidence, terrain, confidence score, model artifacts, and scientific observations. Do not present generated imagery as ground truth. Return a JSON object matching the requested schema; make every field concise and Markdown-friendly for display."""
        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        started = time.perf_counter()
        LOGGER.info("Gemini image analysis: model=%s image_bytes=%d mime_type=%s analysis_type=%s", self.settings.model, len(image_bytes), mime_type, analysis_type)
        try:
            client = self._gemini_client()
            response = client.models.generate_content(
                model=self.settings.model,
                contents=[image_part, prompt, json.dumps({"analysis_type": analysis_type, "metadata": metadata})],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, response_mime_type="application/json",
                    response_schema=SCHEMA, temperature=0.2, max_output_tokens=1800,
                ),
            )
            if DEBUG_ENABLED:
                LOGGER.debug("Gemini SDK response metadata: %s", {"has_text": bool(response.text), "model": self.settings.model, "usage": getattr(getattr(response, "usage_metadata", None), "total_token_count", None)})
            response_text = response.text
            report = json.loads(response_text)
            missing = set(SCHEMA["required"]) - set(report)
            if missing:
                raise ValueError("missing required report fields")
        except (genai_errors.APIError, requests.RequestException, TimeoutError) as error:
            mapped = self._gemini_exception(error)
            LOGGER.exception("Gemini image analysis failed: model=%s image_bytes=%d mime_type=%s code=%s", self.settings.model, len(image_bytes), mime_type, mapped.code)
            raise mapped from error
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            LOGGER.exception("Gemini returned invalid structured output: model=%s", self.settings.model)
            raise VisionAnalysisError("Gemini returned an invalid structured analysis.", "MALFORMED_RESPONSE") from error
        except Exception as error:
            mapped = self._gemini_exception(error)
            LOGGER.exception("Unexpected Gemini image-analysis failure: model=%s image_bytes=%d mime_type=%s code=%s", self.settings.model, len(image_bytes), mime_type, mapped.code)
            raise mapped from error
        latency_ms = round((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage_metadata", None)
        LOGGER.info("Gemini image analysis complete: model=%s latency_ms=%d tokens=%s", self.settings.model, latency_ms, getattr(usage, "total_token_count", None))
        return {"report": report, "provider": "gemini", "model": self.settings.model, "cached": False, "debug": self._debug(model=self.settings.model, endpoint="sdk.generate_content", latency_ms=latency_ms, image_size=len(image_bytes), mime_type=mime_type, status="success", token_usage=getattr(usage, "total_token_count", None))}
