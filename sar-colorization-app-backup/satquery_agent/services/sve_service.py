"""Thread-safe lifecycle, caching, and safe evidence helpers for SVE v1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, Optional, Sequence, TypeVar

from PIL import Image

from ..config.sve_scene_prompts import (
    GROUNDING_RELATED_LABELS,
    SCENE_LABELS,
    VQA_CONCEPT_LABELS,
    prompts_for,
)
from ..models import (
    SVECaptionConsistency,
    SVEGroundingSupport,
    SVEResult,
    SVEScenePrior,
    SVESemanticComparison,
    SVEVQAConsistency,
    SpecialistHealth,
)
from ..satquery_vision_encoder import SatQueryVisionEncoder
from ..sve_artifacts import (
    EXPECTED_ADAPTER_SHA256,
    SVEArtifactMissing,
    SVEChecksumMismatch,
    SVEError,
    SVEInferenceFailure,
    SVELoadFailure,
    verify_sve_artifacts,
)


T = TypeVar("T")
LIMITATIONS = [
    "Adaptation used a geographically limited Lithuania subset.",
    "Adaptation imagery is summer-only.",
    "Version 1 supports RGB and scientifically approved RGB-like optical imagery only.",
    "Evidence is scene-level and does not localize pixels or objects.",
    "Similarity scores are not calibrated probabilities and are not ground truth.",
    "Validation retrieval improved over generic OpenCLIP on the recorded split but remains low in absolute terms.",
]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return min(maximum, max(minimum, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


class BoundedLRU(Generic[T]):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._values: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def put(self, key: str, value: T) -> None:
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


@dataclass(frozen=True)
class SVECall:
    result: SVEResult
    trace: list[dict[str, Any]] = field(default_factory=list)


class SVEManager:
    def __init__(self, encoder_factory: Callable[..., SatQueryVisionEncoder] = SatQueryVisionEncoder) -> None:
        self._encoder_factory = encoder_factory
        self._load_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._encoder: Optional[SatQueryVisionEncoder] = None
        self._artifacts: Any = None
        self._state = "disabled" if not self.enabled else "unloaded"
        self._safe_error: Optional[str] = None
        self._load_count = 0
        self._reuse_count = 0
        self._inference_count = 0
        self._failure_count = 0
        self._checksum_failure_count = 0
        self._cpu_fallback_count = 0
        self._mps_fallback_count = 0
        self._caption_reranking_count = 0
        self._vqa_agreement_count = 0
        self._vqa_disagreement_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._runtime_total_ms = 0
        self._device_usage: Counter[str] = Counter()
        capacity = _int_env("SVE_CACHE_SIZE", 128, 1, 2048)
        self._image_cache: BoundedLRU[Any] = BoundedLRU(capacity)
        self._text_cache: BoundedLRU[Any] = BoundedLRU(capacity)
        self._prompt_cache: BoundedLRU[Any] = BoundedLRU(len(SCENE_LABELS))

    @property
    def enabled(self) -> bool:
        return _bool_env("SVE_ENABLED", True)

    @property
    def max_batch_size(self) -> int:
        return _int_env("SVE_MAX_BATCH_SIZE", 8, 1, 64)

    @property
    def timeout_seconds(self) -> int:
        return _int_env("SVE_TIMEOUT_SECONDS", 30, 1, 600)

    @property
    def state(self) -> str:
        return self._state

    def _select_device(self) -> str:
        import torch

        requested = os.getenv("SVE_DEVICE", "auto").strip().lower()
        if requested not in {"auto", "cuda", "mps", "cpu"}:
            raise SVELoadFailure("SVE_DEVICE is unsupported")
        cuda = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps = bool(mps_backend is not None and mps_backend.is_available())
        if requested == "cuda":
            if not cuda:
                raise SVELoadFailure("Configured CUDA device is unavailable")
            return "cuda"
        if requested == "mps":
            if not mps:
                raise SVELoadFailure("Configured MPS device is unavailable")
            return "mps"
        if requested == "cpu":
            return "cpu"
        return "cuda" if cuda else "mps" if mps else "cpu"

    def health(self) -> SpecialistHealth:
        return SpecialistHealth(
            status=self._state,
            device=self._encoder.device if self._encoder is not None else None,
            error=self._safe_error,
        )

    def load(self) -> tuple[int, bool]:
        if not self.enabled:
            self._state = "disabled"
            raise SVEArtifactMissing("SVE is disabled")
        if self._state == "ready" and self._encoder is not None:
            with self._metrics_lock:
                self._reuse_count += 1
            return 0, True
        with self._load_lock:
            if self._state == "ready" and self._encoder is not None:
                with self._metrics_lock:
                    self._reuse_count += 1
                return 0, True
            if self._state == "failed":
                raise SVELoadFailure("SVE remains failed until explicit retry")
            started = time.perf_counter()
            try:
                self._state = "verifying"
                artifacts = verify_sve_artifacts()
                self._state = "loading"
                encoder = self._encoder_factory(artifacts, self._select_device())
                encoder.load()
                self._artifacts = artifacts
                self._encoder = encoder
                self._state = "ready"
                self._safe_error = None
                with self._metrics_lock:
                    self._load_count += 1
                return max(0, round((time.perf_counter() - started) * 1000)), False
            except SVEError as error:
                self._state = "failed"
                self._safe_error = error.public_message
                with self._metrics_lock:
                    self._failure_count += 1
                    if isinstance(error, SVEChecksumMismatch):
                        self._checksum_failure_count += 1
                raise
            except Exception as error:
                self._state = "failed"
                self._safe_error = SVELoadFailure.public_message
                with self._metrics_lock:
                    self._failure_count += 1
                raise SVELoadFailure("Unexpected model load failure") from error

    def retry(self) -> None:
        with self._load_lock:
            if self._encoder is not None:
                self._encoder.close()
            self._encoder = None
            self._artifacts = None
            self._safe_error = None
            self._state = "unloaded" if self.enabled else "disabled"
            self._image_cache.clear()
            self._text_cache.clear()
            self._prompt_cache.clear()

    def _identity(self) -> str:
        if self._artifacts is None:
            raise SVELoadFailure("Verified artifacts are unavailable")
        preprocessing = json.dumps(self._artifacts.preprocessing, sort_keys=True, separators=(",", ":"))
        preprocessing_version = hashlib.sha256(preprocessing.encode("utf-8")).hexdigest()[:16]
        return f"{EXPECTED_ADAPTER_SHA256}|ViT-L-14|laion2b_s32b_b82k|{preprocessing_version}"

    def _cache_record(self, hit: bool) -> None:
        with self._metrics_lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def _run_with_mps_fallback(self, operation: Callable[[], T]) -> tuple[T, Optional[str]]:
        if self._encoder is None:
            raise SVELoadFailure("Encoder is unavailable")
        try:
            return operation(), None
        except SVEInferenceFailure:
            if self._encoder.device != "mps":
                raise
            self._encoder.move_to("cpu")
            with self._metrics_lock:
                self._mps_fallback_count += 1
                self._cpu_fallback_count += 1
            return operation(), "MPS inference failed for an unsupported operation; execution retried on CPU."

    def _image_embedding(self, image: Image.Image, content_hash: str) -> tuple[Any, bool, Optional[str]]:
        if image.width <= 0 or image.height <= 0 or image.width > 65535 or image.height > 65535:
            raise SVEInferenceFailure("Image dimensions are invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash.lower()):
            raise SVEInferenceFailure("A SHA-256 image content hash is required")
        key = hashlib.sha256(f"image|{content_hash}|{self._identity()}".encode("utf-8")).hexdigest()
        cached = self._image_cache.get(key)
        if cached is not None:
            self._cache_record(True)
            return cached.clone(), True, None
        self._cache_record(False)
        if self._encoder is None:
            raise SVELoadFailure("Encoder is unavailable")
        embedding, fallback = self._run_with_mps_fallback(lambda: self._encoder.encode_image(image))
        self._image_cache.put(key, embedding.clone())
        return embedding, False, fallback

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = " ".join(text.strip().split())
        if not normalized or len(normalized) > 500:
            raise SVEInferenceFailure("Text input must contain 1 to 500 characters")
        return normalized

    def _text_embedding(self, texts: Sequence[str]) -> tuple[Any, int, Optional[str]]:
        if len(texts) > self.max_batch_size:
            raise SVEInferenceFailure("Text batch exceeds the configured SVE maximum")
        if self._encoder is None:
            raise SVELoadFailure("Encoder is unavailable")
        import torch

        normalized = [self._normalize_text(text) for text in texts]
        outputs: list[Any] = [None] * len(normalized)
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        hits = 0
        for index, text in enumerate(normalized):
            key = hashlib.sha256(f"text|{text}|{self._identity()}".encode("utf-8")).hexdigest()
            cached = self._text_cache.get(key)
            if cached is None:
                self._cache_record(False)
                missing_texts.append(text)
                missing_indices.append(index)
            else:
                self._cache_record(True)
                outputs[index] = cached.clone()
                hits += 1
        fallback: Optional[str] = None
        if missing_texts:
            encoded, fallback = self._run_with_mps_fallback(lambda: self._encoder.encode_text(missing_texts))
            for row, index, text in zip(encoded, missing_indices, missing_texts):
                value = row.unsqueeze(0).clone()
                outputs[index] = value
                key = hashlib.sha256(f"text|{text}|{self._identity()}".encode("utf-8")).hexdigest()
                self._text_cache.put(key, value)
        return torch.cat(outputs, dim=0), hits, fallback

    def encode_shared_features(
        self,
        image: Image.Image,
        text: str,
        *,
        content_hash: Optional[str] = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Encode an image/question pair with the one cached SVE OpenCLIP instance.

        This is the narrow public bridge used by learned heads that were trained
        on the frozen SatQuery Vision Encoder.  Embeddings remain detached CPU
        tensors so callers cannot mutate or move the shared backbone.
        """
        started = time.perf_counter()
        load_ms, reused = self.load()
        if content_hash is None:
            digest = hashlib.sha256()
            rgb = image.convert("RGB")
            digest.update(f"{rgb.width}x{rgb.height}|RGB|".encode("ascii"))
            digest.update(rgb.tobytes())
            content_hash = digest.hexdigest()
        image_embedding, image_cache_hit, image_fallback = self._image_embedding(
            image, content_hash
        )
        text_embedding, text_cache_hits, text_fallback = self._text_embedding([text])
        runtime_ms = max(0, round((time.perf_counter() - started) * 1000))
        self._record_inference(runtime_ms)
        return (
            image_embedding.detach().cpu(),
            text_embedding.detach().cpu(),
            {
                "device": self._encoder.device if self._encoder is not None else None,
                "runtime_ms": runtime_ms,
                "model_load_ms": load_ms,
                "model_reused": reused,
                "image_cache_hit": image_cache_hit,
                "text_cache_hit": text_cache_hits == 1,
                "fallback": " ".join(
                    value for value in (image_fallback, text_fallback) if value
                ) or None,
            },
        )

    def _prompt_embedding(self, label: str) -> Any:
        import torch

        key = hashlib.sha256(f"prompt|{label}|{self._identity()}".encode("utf-8")).hexdigest()
        cached = self._prompt_cache.get(key)
        if cached is not None:
            self._cache_record(True)
            return cached.clone()
        texts = prompts_for(label)
        embeddings, _, _ = self._text_embedding(texts)
        averaged = embeddings.mean(dim=0, keepdim=True)
        averaged = averaged / averaged.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
        self._prompt_cache.put(key, averaged.clone())
        return averaged

    def _all_priors(self, image_embedding: Any) -> list[SVEScenePrior]:
        if self._encoder is None:
            raise SVELoadFailure("Encoder is unavailable")
        prompt_embeddings = [self._prompt_embedding(label) for label in SCENE_LABELS]
        import torch

        matrix = torch.cat(prompt_embeddings, dim=0)
        scores = self._encoder.cosine_similarity(image_embedding, matrix)[0]
        values = [
            SVEScenePrior(label=label, similarity=round(float(score), 6))
            for label, score in zip(SCENE_LABELS, scores)
        ]
        return sorted(values, key=lambda item: (-item.similarity, item.label))

    def _caption_consistency(self, image_embedding: Any, candidates: Sequence[str]) -> SVECaptionConsistency:
        if not candidates or len(candidates) > self.max_batch_size:
            raise SVEInferenceFailure("Caption candidate batch is invalid")
        if self._encoder is None:
            raise SVELoadFailure("Encoder is unavailable")
        embeddings, _, _ = self._text_embedding(candidates)
        scores = [round(float(value), 6) for value in self._encoder.cosine_similarity(image_embedding, embeddings)[0]]
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        if len(candidates) > 1:
            with self._metrics_lock:
                self._caption_reranking_count += 1
        return SVECaptionConsistency(
            score=scores[ranked[0]],
            selected_candidate_index=ranked[0],
            original_candidates=list(candidates),
            original_candidate_order=list(range(len(candidates))),
            reranked_candidate_order=ranked,
            candidate_scores=scores,
            reranked=len(candidates) > 1 and ranked != list(range(len(candidates))),
        )

    def _vqa_consistency(
        self,
        category: Optional[str],
        answer: Optional[str],
        all_priors: Sequence[SVEScenePrior],
    ) -> Optional[SVEVQAConsistency]:
        labels = VQA_CONCEPT_LABELS.get(category or "")
        if not labels or not answer:
            return None
        ranks = {prior.label: index for index, prior in enumerate(all_priors)}
        candidates = [prior for prior in all_priors if prior.label in labels]
        if not candidates:
            return SVEVQAConsistency(state="unavailable", explanation="No controlled scene-prior mapping was available.")
        strongest = max(candidates, key=lambda item: item.similarity)
        rank = min(ranks[item.label] for item in candidates)
        normalized = answer.strip().lower()
        negative = normalized.startswith("no") or any(term in normalized for term in (" not visible", "not detected", "does not", "unavailable"))
        if negative:
            state = "agreement" if rank >= 5 else "disagreement" if rank <= 2 else "weak agreement"
        else:
            state = "agreement" if rank <= 2 else "weak agreement" if rank <= 6 else "disagreement"
        explanation = (
            "The VQA answer and scene-level embedding evidence disagree."
            if state == "disagreement"
            else "The VQA answer has supporting scene-level embedding evidence."
            if state == "agreement"
            else "The VQA answer has limited scene-level embedding support."
        )
        with self._metrics_lock:
            if state == "disagreement":
                self._vqa_disagreement_count += 1
            elif state == "agreement":
                self._vqa_agreement_count += 1
        return SVEVQAConsistency(
            state=state,
            target_concept=strongest.label,
            similarity=strongest.similarity,
            explanation=explanation,
        )

    @staticmethod
    def _grounding_support(target: Optional[str], all_priors: Sequence[SVEScenePrior]) -> Optional[SVEGroundingSupport]:
        if not target:
            return None
        normalized = target.strip().lower()
        related: tuple[str, ...] = ()
        for key, labels in GROUNDING_RELATED_LABELS.items():
            if key in normalized:
                related = labels
                break
        if not related:
            return SVEGroundingSupport(state="unavailable", target=target, related_scene_labels=[])
        ranks = {prior.label: index for index, prior in enumerate(all_priors)}
        best_rank = min((ranks[label] for label in related if label in ranks), default=len(all_priors))
        state = "supported" if best_rank <= 3 else "limited" if best_rank >= 8 else "weak"
        warning = (
            f"Scene-level evidence for a {normalized}-related environment is limited."
            if state == "limited" else None
        )
        return SVEGroundingSupport(
            state=state,
            target=target,
            related_scene_labels=[label for label in related if label in ranks],
            warning=warning,
        )

    def _record_inference(self, runtime_ms: int) -> None:
        if self._encoder is None:
            return
        with self._metrics_lock:
            self._inference_count += 1
            self._runtime_total_ms += runtime_ms
            self._device_usage[self._encoder.device] += 1

    def _success_result(self, runtime_ms: int, **updates: Any) -> SVEResult:
        device = self._encoder.device if self._encoder is not None else None
        fallback = updates.pop("fallback", None)
        return SVEResult(
            available=True,
            status="success",
            device=device,
            runtime_ms=runtime_ms,
            limitations=list(LIMITATIONS),
            fallback=fallback,
            **updates,
        )

    def unavailable_result(self, error: SVEError | None = None, status: Optional[str] = None) -> SVEResult:
        if status is None:
            if not self.enabled:
                status = "disabled"
            elif isinstance(error, SVEChecksumMismatch):
                status = "checksum_failure"
            elif isinstance(error, SVEArtifactMissing):
                status = "unavailable"
            else:
                status = "model_error"
        return SVEResult(
            available=False,
            status=status,
            limitations=list(LIMITATIONS),
            warning=error.public_message if error else self._safe_error,
        )

    def analyze(
        self,
        image: Image.Image,
        content_hash: str,
        *,
        captions: Sequence[str] = (),
        vqa_category: Optional[str] = None,
        vqa_answer: Optional[str] = None,
        grounding_target: Optional[str] = None,
        top_k: int = 5,
    ) -> SVECall:
        traces: list[dict[str, Any]] = []
        started = time.perf_counter()
        try:
            load_ms, reused = self.load()
            traces.extend([
                {"tool": "sve_artifact_verification", "status": "success", "duration_ms": 0, "parameters": {"checksum_verified": True}},
                {"tool": "sve_model_reuse" if reused else "sve_model_load", "status": "success", "duration_ms": load_ms, "parameters": {"device": self._encoder.device if self._encoder else None, "reused": reused}},
                {"tool": "sve_eligibility_check", "status": "success", "duration_ms": 0, "parameters": {"input": "optical_rgb_like"}},
                {"tool": "sve_image_preparation", "status": "success", "duration_ms": 0, "parameters": {"transform": "verified_openclip_validation_transform"}},
            ])
            embedding_started = time.perf_counter()
            device_before_inference = self._encoder.device if self._encoder else None
            image_embedding, image_cache_hit, fallback = self._image_embedding(image, content_hash)
            traces.append({"tool": "sve_image_embedding", "status": "success", "duration_ms": max(0, round((time.perf_counter() - embedding_started) * 1000)), "parameters": {"cache": "hit" if image_cache_hit else "miss", "device": self._encoder.device if self._encoder else None}})
            prior_started = time.perf_counter()
            all_priors = self._all_priors(image_embedding)
            if device_before_inference == "mps" and self._encoder and self._encoder.device == "cpu" and not fallback:
                fallback = "MPS inference failed for an unsupported operation; execution retried on CPU."
            priors = all_priors[:max(1, min(len(SCENE_LABELS), top_k))]
            traces.append({"tool": "sve_scene_priors", "status": "success", "duration_ms": max(0, round((time.perf_counter() - prior_started) * 1000)), "parameters": {"top_labels": [item.label for item in priors], "calibrated": False}})
            traces.append({"tool": "sve_routing_support", "status": "success", "duration_ms": 0, "parameters": {"support_state": "advisory", "influence_applied": False}})
            caption_consistency = self._caption_consistency(image_embedding, captions) if captions else None
            if caption_consistency is not None:
                traces.append({"tool": "sve_caption_reranking" if len(captions) > 1 else "sve_caption_consistency", "status": "success", "duration_ms": 0, "parameters": {"candidate_count": len(captions), "selected_candidate_index": caption_consistency.selected_candidate_index}})
            vqa_consistency = self._vqa_consistency(vqa_category, vqa_answer, all_priors)
            if vqa_consistency is not None:
                traces.append({"tool": "sve_vqa_consistency", "status": "success", "duration_ms": 0, "parameters": {"consistency_state": vqa_consistency.state}})
            grounding_support = self._grounding_support(grounding_target, all_priors)
            if grounding_support is not None:
                traces.append({"tool": "sve_grounding_support", "status": "success", "duration_ms": 0, "parameters": {"support_state": grounding_support.state}})
            runtime_ms = max(0, round((time.perf_counter() - started) * 1000))
            if runtime_ms > self.timeout_seconds * 1000:
                fallback = (fallback + " " if fallback else "") + "SVE exceeded its configured advisory timeout."
            self._record_inference(runtime_ms)
            return SVECall(
                result=self._success_result(
                    runtime_ms,
                    scene_priors=priors,
                    caption_consistency=caption_consistency,
                    vqa_consistency=vqa_consistency,
                    grounding_support=grounding_support,
                    fallback=fallback,
                ),
                trace=traces,
            )
        except SVEError as error:
            if self._state != "failed":
                with self._metrics_lock:
                    self._failure_count += 1
            return SVECall(
                result=self.unavailable_result(error),
                trace=traces + [{"tool": "sve_fallback", "status": "failed", "duration_ms": max(0, round((time.perf_counter() - started) * 1000)), "parameters": {"reason": "sve_unavailable"}}],
            )

    def compare(
        self,
        first_image: Image.Image,
        first_hash: str,
        second_image: Image.Image,
        second_hash: str,
        *,
        label: str,
        disclaimer: str,
        top_k: int = 5,
    ) -> SVECall:
        started = time.perf_counter()
        traces: list[dict[str, Any]] = []
        try:
            load_ms, reused = self.load()
            traces.append({"tool": "sve_model_reuse" if reused else "sve_model_load", "status": "success", "duration_ms": load_ms, "parameters": {"device": self._encoder.device if self._encoder else None, "reused": reused}})
            first, first_hit, first_fallback = self._image_embedding(first_image, first_hash)
            second, second_hit, second_fallback = self._image_embedding(second_image, second_hash)
            if self._encoder is None:
                raise SVELoadFailure("Encoder is unavailable")
            similarity = round(float(self._encoder.cosine_similarity(first, second)[0, 0]), 6)
            first_priors = self._all_priors(first)
            second_priors = self._all_priors(second)
            first_by_label = {item.label: item.similarity for item in first_priors}
            second_by_label = {item.label: item.similarity for item in second_priors}
            changes = sorted(
                (
                    {"label": concept, "before_similarity": first_by_label[concept], "after_similarity": second_by_label[concept], "difference": round(second_by_label[concept] - first_by_label[concept], 6)}
                    for concept in SCENE_LABELS
                ),
                key=lambda item: (-abs(item["difference"]), item["label"]),
            )[:top_k]
            fallback = " ".join(value for value in (first_fallback, second_fallback) if value) or None
            runtime_ms = max(0, round((time.perf_counter() - started) * 1000))
            self._record_inference(runtime_ms)
            traces.extend([
                {"tool": "sve_image_embedding", "status": "success", "duration_ms": runtime_ms, "parameters": {"first_cache": "hit" if first_hit else "miss", "second_cache": "hit" if second_hit else "miss", "device": self._encoder.device}},
                {"tool": "sve_scene_priors", "status": "success", "duration_ms": 0, "parameters": {"comparison": True, "calibrated": False}},
            ])
            return SVECall(
                result=self._success_result(
                    runtime_ms,
                    scene_priors=first_priors[:top_k],
                    semantic_comparison=SVESemanticComparison(
                        label=label,
                        status="supporting_evidence",
                        similarity=similarity,
                        prior_changes=changes,
                        disclaimer=disclaimer,
                    ),
                    fallback=fallback,
                ),
                trace=traces,
            )
        except SVEError as error:
            if self._state != "failed":
                with self._metrics_lock:
                    self._failure_count += 1
            return SVECall(
                result=self.unavailable_result(error),
                trace=traces + [{"tool": "sve_fallback", "status": "failed", "duration_ms": max(0, round((time.perf_counter() - started) * 1000)), "parameters": {"reason": "sve_unavailable"}}],
            )

    def retrieve(
        self,
        query: str,
        candidates: Sequence[tuple[str, Image.Image, str]],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rank bounded internal evidence IDs without retaining images or exporting embeddings."""
        started = time.perf_counter()
        if not candidates or len(candidates) > self.max_batch_size:
            raise SVEInferenceFailure("Retrieval candidate batch is invalid")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", evidence_id) for evidence_id, _, _ in candidates):
            raise SVEInferenceFailure("Retrieval evidence IDs must be opaque and bounded")
        self.load()
        text, _, _ = self._text_embedding([query])
        embeddings = []
        for _, image, content_hash in candidates:
            embedding, _, _ = self._image_embedding(image, content_hash)
            embeddings.append(embedding)
        if self._encoder is None:
            raise SVELoadFailure("Encoder is unavailable")
        import torch

        scores = self._encoder.cosine_similarity(text, torch.cat(embeddings, dim=0))[0]
        ranked = sorted(
            ({"evidence_id": evidence_id, "similarity": round(float(score), 6)} for (evidence_id, _, _), score in zip(candidates, scores)),
            key=lambda item: (-item["similarity"], item["evidence_id"]),
        )
        self._record_inference(max(0, round((time.perf_counter() - started) * 1000)))
        return ranked[:max(1, min(top_k, len(ranked)))]

    def unsupported_raw_sar_result(self) -> SVEResult:
        return SVEResult(
            available=False,
            status="unsupported_input",
            limitations=list(LIMITATIONS),
            semantic_comparison=SVESemanticComparison(
                label="Optical-to-generated-RGB semantic consistency",
                status="skipped_raw_sar",
                similarity=None,
                disclaimer="Raw SAR is not encoded. A validated generated RGB-like product is required for this supporting comparison.",
            ),
            warning="SVE skipped the raw SAR input; the existing cross-modal specialist remains unchanged.",
        )

    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            lookups = self._cache_hits + self._cache_misses
            return {
                "enabled": self.enabled,
                "lifecycle_status": self._state,
                "model_load_count": self._load_count,
                "model_reuse_count": self._reuse_count,
                "inference_count": self._inference_count,
                "failure_count": self._failure_count,
                "checksum_failure_count": self._checksum_failure_count,
                "cpu_fallback_count": self._cpu_fallback_count,
                "mps_fallback_count": self._mps_fallback_count,
                "caption_reranking_count": self._caption_reranking_count,
                "vqa_agreement_count": self._vqa_agreement_count,
                "vqa_disagreement_count": self._vqa_disagreement_count,
                "average_runtime_ms": round(self._runtime_total_ms / self._inference_count, 1) if self._inference_count else None,
                "cache_hit_rate_percent": round(self._cache_hits * 100 / lookups, 1) if lookups else None,
                "device_usage": dict(self._device_usage),
            }


SVE_SERVICE = SVEManager()


def sve_configured() -> bool:
    return _bool_env("SVE_ENABLED", True) and all(
        importlib.util.find_spec(name) is not None for name in ("torch", "open_clip", "PIL")
    )


def get_sve_service() -> SVEManager:
    return SVE_SERVICE
