"""Exact inference reconstruction for the verified SatQuery Vision Encoder v1 adapter."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from .sve_artifacts import (
    EXPECTED_EMBEDDING_DIMENSION,
    EXPECTED_OPENCLIP_MODEL,
    EXPECTED_PRETRAINED,
    SVEInferenceFailure,
    SVELoadFailure,
    VerifiedSVEArtifacts,
)


ADAPTER_PREFIX = "clip_model."
UNFROZEN_VISUAL_BLOCKS = 2


def _expected_adapter_keys(model_keys: Iterable[str]) -> set[str]:
    keys = set(model_keys)
    selected = {
        key for key in keys
        if key in {"logit_scale", "visual.proj"}
        or key.startswith("visual.transformer.resblocks.22.")
        or key.startswith("visual.transformer.resblocks.23.")
        or key.startswith("visual.ln_post.")
    }
    return selected


class SatQueryVisionEncoder:
    """Private model wrapper; callers receive normalized tensors, never the model object."""

    def __init__(
        self,
        artifacts: VerifiedSVEArtifacts,
        device: str,
        *,
        model_factory: Callable[..., Any] | None = None,
        tokenizer_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.device = device
        self._model_factory = model_factory
        self._tokenizer_factory = tokenizer_factory
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        try:
            import torch
            if self._model_factory is None or self._tokenizer_factory is None:
                import open_clip

                model_factory = self._model_factory or open_clip.create_model_and_transforms
                tokenizer_factory = self._tokenizer_factory or open_clip.get_tokenizer
            else:
                model_factory = self._model_factory
                tokenizer_factory = self._tokenizer_factory

            created = model_factory(
                EXPECTED_OPENCLIP_MODEL,
                pretrained=EXPECTED_PRETRAINED,
                device="cpu",
            )
            if not isinstance(created, tuple) or len(created) != 3:
                raise SVELoadFailure("OpenCLIP factory returned an unexpected value")
            model, _, preprocess = created
            payload = torch.load(self.artifacts.adapter_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or not isinstance(payload.get("adapter_state_dict"), dict):
                raise SVELoadFailure("Adapter payload is malformed")
            required_metadata = {
                "model_name": "SatQuery Vision Encoder v1",
                "backbone": EXPECTED_OPENCLIP_MODEL,
                "pretrained": EXPECTED_PRETRAINED,
                "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
                "unfreeze_visual_blocks": UNFROZEN_VISUAL_BLOCKS,
                "adaptation_dataset": "BigEarthNet.txt",
                "image_dataset": "BigEarthNet v2 Lithuania Summer",
                "train_samples": 4008,
                "validation_samples": 2291,
                "test_samples": 2053,
                "epoch": 5,
            }
            if any(payload.get(key) != value for key, value in required_metadata.items()):
                raise SVELoadFailure("Adapter training metadata does not match the verified reconstruction")

            raw_state = payload["adapter_state_dict"]
            if any(not isinstance(key, str) or not key.startswith(ADAPTER_PREFIX) for key in raw_state):
                raise SVELoadFailure("Adapter contains an unknown parameter namespace")
            adapted = {key[len(ADAPTER_PREFIX):]: value for key, value in raw_state.items()}
            model_state = model.state_dict()
            expected = _expected_adapter_keys(model_state)
            actual = set(adapted)
            if actual != expected:
                missing = sorted(expected - actual)
                unknown = sorted(actual - expected)
                raise SVELoadFailure(
                    f"Adapter key validation failed ({len(missing)} missing, {len(unknown)} unknown)"
                )
            for key, value in adapted.items():
                expected_value = model_state.get(key)
                if expected_value is None or not isinstance(value, torch.Tensor):
                    raise SVELoadFailure("Adapter contains an invalid tensor")
                if tuple(value.shape) != tuple(expected_value.shape):
                    raise SVELoadFailure("Adapter tensor shape does not match the exact backbone")

            incompatible = model.load_state_dict(adapted, strict=False)
            if incompatible.unexpected_keys:
                raise SVELoadFailure("Adapter contains unexpected parameters")
            if set(model_state) - set(incompatible.missing_keys) != expected:
                raise SVELoadFailure("Adapter application did not affect exactly the verified parameters")
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.to(self.device, dtype=torch.float32)

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer_factory(EXPECTED_OPENCLIP_MODEL)
        except SVELoadFailure:
            self.close()
            raise
        except Exception as error:
            self.close()
            raise SVELoadFailure("Exact OpenCLIP reconstruction failed") from error

    def close(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def move_to(self, device: str) -> None:
        if self._model is None:
            raise SVELoadFailure("Encoder is not loaded")
        try:
            import torch

            self._model.to(device, dtype=torch.float32)
            self.device = device
            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
        except Exception as error:
            raise SVELoadFailure("Encoder device transfer failed") from error

    @staticmethod
    def _normalize(embedding: Any) -> Any:
        import torch

        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
            raise SVEInferenceFailure("Encoder produced an invalid embedding tensor")
        if embedding.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
            raise SVEInferenceFailure("Encoder produced an unexpected embedding dimension")
        embedding = embedding.float()
        norms = embedding.norm(dim=-1, keepdim=True)
        if not torch.isfinite(embedding).all() or not torch.isfinite(norms).all() or torch.any(norms <= 0):
            raise SVEInferenceFailure("Encoder produced a non-finite embedding")
        return embedding / norms

    def encode_image(self, images: Image.Image | Sequence[Image.Image]) -> Any:
        if self._model is None or self._preprocess is None:
            raise SVELoadFailure("Encoder is not loaded")
        import torch

        batch = [images] if isinstance(images, Image.Image) else list(images)
        if not batch:
            raise SVEInferenceFailure("At least one image is required")
        try:
            prepared = torch.stack([self._preprocess(image.convert("RGB")) for image in batch]).to(
                self.device, dtype=torch.float32
            )
            with torch.inference_mode():
                embedding = self._model.encode_image(prepared)
            return self._normalize(embedding).detach().cpu()
        except (SVELoadFailure, SVEInferenceFailure):
            raise
        except Exception as error:
            raise SVEInferenceFailure("Image embedding inference failed") from error

    def encode_text(self, texts: str | Sequence[str]) -> Any:
        if self._model is None or self._tokenizer is None:
            raise SVELoadFailure("Encoder is not loaded")
        import torch

        values = [texts] if isinstance(texts, str) else list(texts)
        if not values or any(not isinstance(text, str) or not text.strip() for text in values):
            raise SVEInferenceFailure("Non-empty text is required")
        try:
            tokens = self._tokenizer(values).to(self.device)
            with torch.inference_mode():
                embedding = self._model.encode_text(tokens)
            return self._normalize(embedding).detach().cpu()
        except (SVELoadFailure, SVEInferenceFailure):
            raise
        except Exception as error:
            raise SVEInferenceFailure("Text embedding inference failed") from error

    @staticmethod
    def cosine_similarity(first: Any, second: Any) -> Any:
        import torch

        scores = first.float() @ second.float().T
        if not torch.isfinite(scores).all():
            raise SVEInferenceFailure("Similarity computation was non-finite")
        return scores.clamp(-1, 1)

    def image_text_similarity(self, image: Image.Image, texts: Sequence[str]) -> list[float]:
        image_embedding = self.encode_image(image)
        text_embedding = self.encode_text(texts)
        return [float(value) for value in self.cosine_similarity(image_embedding, text_embedding)[0]]

    def rank_text_candidates(self, image: Image.Image, candidates: Sequence[str]) -> list[tuple[int, float]]:
        scores = self.image_text_similarity(image, candidates)
        return sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))

    @staticmethod
    def validate_similarity(value: float) -> float:
        if not math.isfinite(value):
            raise SVEInferenceFailure("Similarity is non-finite")
        return max(-1.0, min(1.0, float(value)))
