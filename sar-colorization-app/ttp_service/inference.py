"""Official OpenCDInferencer adapter using TTP's bundled Python packages."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class TTPInferenceError(RuntimeError):
    pass


def _unwrap_prediction(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "data") and value.data is not value:
        found = _unwrap_prediction(value.data)
        if found is not None:
            return found
    if hasattr(value, "pred_sem_seg"):
        found = _unwrap_prediction(value.pred_sem_seg)
        if found is not None:
            return found
    if isinstance(value, dict):
        for key in ("predictions", "prediction", "pred_sem_seg", "data_samples"):
            if key in value:
                found = _unwrap_prediction(value[key])
                if found is not None:
                    return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _unwrap_prediction(item)
            if found is not None:
                return found
    return None


def _unwrap_logits(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "seg_logits"):
        found = _unwrap_prediction(value.seg_logits)
        if found is not None:
            return found
    if isinstance(value, dict):
        for key in ("seg_logits", "logits", "data_samples", "predictions"):
            if key in value:
                found = _unwrap_logits(value[key]) if key in {"data_samples", "predictions"} else _unwrap_prediction(value[key])
                if found is not None:
                    return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _unwrap_logits(item)
            if found is not None:
                return found
    return None


def _softmax_changed_probability(logits: np.ndarray, class_axis: int) -> np.ndarray:
    stable = logits.astype(np.float64, copy=False) - np.max(logits, axis=class_axis, keepdims=True)
    exponent = np.exp(stable)
    probabilities = exponent / np.sum(exponent, axis=class_axis, keepdims=True)
    return np.take(probabilities, 1, axis=class_axis).astype(np.float32)


def changed_probability_map(value: Any) -> tuple[np.ndarray, str]:
    """Extract changed-class probability while retaining old binary outputs."""
    logits = _unwrap_logits(value)
    if logits is not None:
        values = np.squeeze(np.asarray(logits))
        if values.ndim == 3 and values.shape[0] == 2:
            probability = _softmax_changed_probability(values, 0)
            source = "seg_logits_softmax"
        elif values.ndim == 3 and values.shape[-1] == 2:
            probability = _softmax_changed_probability(values, -1)
            source = "seg_logits_softmax"
        elif values.ndim == 2:
            if float(values.min()) >= 0.0 and float(values.max()) <= 1.0:
                probability = values.astype(np.float32)
                source = "changed_probability"
            else:
                probability = (1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))).astype(np.float32)
                source = "changed_logit_sigmoid"
        else:
            raise TTPInferenceError("TTP returned unsupported segmentation logits.")
    else:
        prediction = _unwrap_prediction(value)
        if prediction is None:
            raise TTPInferenceError("TTP returned no binary prediction or probability map.")
        prediction = np.squeeze(np.asarray(prediction))
        if prediction.ndim == 3 and prediction.shape[0] == 2:
            prediction = np.argmax(prediction, axis=0)
        prediction = np.squeeze(prediction)
        if prediction.ndim != 2:
            raise TTPInferenceError("TTP returned an invalid prediction shape.")
        unique = set(np.unique(prediction).tolist())
        if unique.issubset({0, 1, False, True}):
            probability = prediction.astype(np.float32)
            source = "binary_prediction_compatibility"
        elif np.issubdtype(prediction.dtype, np.floating) and float(prediction.min()) >= 0.0 and float(prediction.max()) <= 1.0:
            probability = prediction.astype(np.float32)
            source = "changed_probability"
        else:
            raise TTPInferenceError("TTP returned values outside the binary/probability range.")
    if probability.ndim != 2 or not np.all(np.isfinite(probability)):
        raise TTPInferenceError("TTP returned an invalid or non-finite probability map.")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise TTPInferenceError("TTP probability values were outside [0,1].")
    return probability, source


class TTPInferenceEngine:
    def __init__(self, repository: Path, checkpoint: Path) -> None:
        self.repository = repository.resolve()
        self.checkpoint = checkpoint.resolve()
        self.config = self.repository / "configs" / "TTP" / "ttp_sam_large_levircd_infer.py"
        self._inferencer: Any = None
        self.last_probability_map: np.ndarray | None = None
        self.last_probability_source: str | None = None

    def load(self) -> None:
        if not self.repository.is_dir() or not self.config.is_file() or not self.checkpoint.is_file():
            raise TTPInferenceError("Verified TTP assets are unavailable.")
        for path in (self.repository / "opencd", self.repository / "mmseg", self.repository):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            from opencd.apis import OpenCDInferencer
            self._inferencer = OpenCDInferencer(
                model=str(self.config),
                weights=str(self.checkpoint),
                classes=("unchanged", "changed"),
                palette=[[0, 0, 0], [255, 255, 255]],
                device="cuda:0",
            )
        except Exception as error:
            raise TTPInferenceError("TTP model initialization failed.") from error

    def predict(self, earlier_data: bytes, later_data: bytes, earlier_suffix: str, later_suffix: str) -> np.ndarray:
        if self._inferencer is None:
            raise TTPInferenceError("TTP model is not loaded.")
        with tempfile.TemporaryDirectory(prefix="ttp-request-") as directory:
            root = Path(directory)
            earlier = root / f"earlier{earlier_suffix}"
            later = root / f"later{later_suffix}"
            earlier.write_bytes(earlier_data)
            later.write_bytes(later_data)
            try:
                result = self._inferencer([[str(earlier), str(later)]], show=False, return_datasamples=True)
            except TypeError:
                result = self._inferencer([[str(earlier), str(later)]], show=False)
        probability, source = changed_probability_map(result)
        self.last_probability_map = probability
        self.last_probability_source = source
        return (probability > 0.5).astype(bool, copy=False)
