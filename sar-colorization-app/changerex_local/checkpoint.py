"""Strict verification and loading for the pinned official checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .architecture import model_parameter_count
from .config import CHECKPOINT_SHA256
from .schemas import CheckpointVerificationReport


class CheckpointError(RuntimeError):
    """Base class for safe checkpoint failures."""


class CheckpointVerificationError(CheckpointError):
    pass


class CheckpointCompatibilityError(CheckpointError):
    pass


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_metadata(value: Any, *, depth: int = 0) -> Any:
    """Produce a JSON-safe, bounded metadata representation."""
    if depth > 4:
        return "<maximum-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "…"
    if isinstance(value, Mapping):
        items = list(value.items())[:100]
        result = {str(key): _bounded_metadata(item, depth=depth + 1) for key, item in items}
        if len(value) > 100:
            result["<truncated-items>"] = len(value) - 100
        return result
    if isinstance(value, (list, tuple)):
        result = [_bounded_metadata(item, depth=depth + 1) for item in value[:100]]
        if len(value) > 100:
            result.append(f"<truncated-items:{len(value) - 100}>")
        return result
    if isinstance(value, torch.Tensor):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    text = repr(value)
    return text if len(text) <= 500 else text[:500] + "…"


def _load_verified_payload(path: Path, expected_sha256: str) -> tuple[dict[str, Any], int, str]:
    if not path.is_file():
        raise CheckpointVerificationError(f"Checkpoint file does not exist: {path}")
    file_size = path.stat().st_size
    digest = sha256_file(path)
    if digest.lower() != expected_sha256.lower():
        raise CheckpointVerificationError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, received {digest}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise CheckpointVerificationError(f"Verified checkpoint could not be decoded: {error}") from error
    if not isinstance(payload, dict):
        raise CheckpointVerificationError("Checkpoint top level must be a dictionary")
    return payload, file_size, digest


def inspect_checkpoint(
    path: str | Path, *, expected_sha256: str = CHECKPOINT_SHA256
) -> tuple[dict[str, torch.Tensor], CheckpointVerificationReport]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload, file_size, digest = _load_verified_payload(checkpoint_path, expected_sha256)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise CheckpointVerificationError("Checkpoint does not contain a non-empty 'state_dict'")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()):
        raise CheckpointVerificationError("Checkpoint state_dict contains non-tensor entries")

    # The official file already uses exact model-relative keys. No prefix
    # normalization is performed; any mismatch is an incompatibility.
    normalized = dict(state_dict)
    metadata = _bounded_metadata(payload.get("meta", {}))
    try:
        json.dumps(metadata)
    except TypeError as error:  # defensive invariant for diagnostics output
        raise CheckpointVerificationError("Checkpoint metadata is not serializable") from error
    report = CheckpointVerificationReport(
        path=str(checkpoint_path),
        file_size=file_size,
        sha256=digest,
        top_level_keys=tuple(sorted(str(key) for key in payload)),
        tensor_count=len(normalized),
        parameter_count=0,
        missing_keys=(),
        unexpected_keys=(),
        key_transformations=(),
        checkpoint_metadata=metadata,
    )
    return normalized, report


def load_checkpoint_strict(
    model: nn.Module,
    path: str | Path,
    *,
    expected_sha256: str = CHECKPOINT_SHA256,
) -> CheckpointVerificationReport:
    state_dict, report = inspect_checkpoint(path, expected_sha256=expected_sha256)
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"Strict state-dict loading failed: {error}") from error
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if missing or unexpected:
        raise CheckpointCompatibilityError(
            f"Strict state-dict loading reported missing={missing}, unexpected={unexpected}"
        )
    return CheckpointVerificationReport(
        path=report.path,
        file_size=report.file_size,
        sha256=report.sha256,
        top_level_keys=report.top_level_keys,
        tensor_count=report.tensor_count,
        parameter_count=model_parameter_count(model),
        missing_keys=missing,
        unexpected_keys=unexpected,
        key_transformations=report.key_transformations,
        checkpoint_metadata=report.checkpoint_metadata,
    )
