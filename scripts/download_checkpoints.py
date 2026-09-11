#!/usr/bin/env python3
"""Download GeoVision checkpoints from a Hugging Face model repository.

The script never overwrites a non-empty checkpoint. Set HF_MODEL_REPO to a
repository such as ``your-account/geovision-checkpoints`` and, for private or
gated repositories, set HF_TOKEN. Destination paths follow the same three
environment variables consumed by ``sar-colorization-app/backend.py``.
"""

from __future__ import annotations

import os
import shutil
import sys
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:  # Allows a clear configuration error before dependencies are installed.
    hf_hub_download = None
    HfHubHTTPError = RuntimeError


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Checkpoint:
    label: str
    local_environment_key: str
    default_local_path: str
    remote_environment_key: str
    default_remote_filename: str


CHECKPOINTS = (
    Checkpoint("Pix2Pix", "PIX2PIX_CHECKPOINT", "pix2pix_gen_180.pth", "PIX2PIX_REMOTE_FILENAME", "pix2pix_gen_180.pth"),
    Checkpoint("SARFusionFormer", "SARFUSIONFORMER_CHECKPOINT", "models/checkpoints/sarfusionformer_256_decoder_best.pt", "SARFUSIONFORMER_REMOTE_FILENAME", "sarfusionformer_256_decoder_best.pt"),
    Checkpoint("Color corrector", "COLOR_CORRECTOR_CHECKPOINT", "models/checkpoints/color_corrector_256_best.pt", "COLOR_CORRECTOR_REMOTE_FILENAME", "color_corrector_256_best.pt"),
)


def resolve_local_path(environment_key: str, default_relative_path: str) -> Path:
    configured = os.getenv(environment_key, default_relative_path).strip()
    candidate = Path(configured).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()


def remote_filename(environment_key: str, default_filename: str) -> str:
    filename = os.getenv(environment_key, default_filename).strip()
    path = PurePosixPath(filename)
    if not filename or path.is_absolute() or ".." in path.parts:
        raise ValueError("{} must be a safe Hugging Face repository path.".format(environment_key))
    return filename


def validate_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("{} download is missing or empty at {}".format(label, path))


def sha256_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(checkpoint: Checkpoint, repository: str, token: Optional[str]) -> None:
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is not installed. Install sar-colorization-app/requirements-backend.txt first."
        )
    destination = resolve_local_path(checkpoint.local_environment_key, checkpoint.default_local_path)
    if destination.is_file() and destination.stat().st_size > 0:
        print("{}: keeping existing checkpoint ({:,} bytes): {}".format(checkpoint.label, destination.stat().st_size, destination))
        return

    filename = remote_filename(checkpoint.remote_environment_key, checkpoint.default_remote_filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("{}: downloading {} from {}".format(checkpoint.label, filename, repository))
    try:
        downloaded = Path(hf_hub_download(repo_id=repository, filename=filename, token=token or None))
    except HfHubHTTPError as error:
        raise RuntimeError(
            "Could not download {} from {}. Check HF_MODEL_REPO, the remote filename, and HF_TOKEN access for private repositories.".format(checkpoint.label, repository)
        ) from error
    except OSError as error:
        raise RuntimeError("Failed to download {}: {}".format(checkpoint.label, error)) from error

    validate_file(downloaded, checkpoint.label)
    downloaded_digest = sha256_digest(downloaded)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        shutil.copyfile(downloaded, temporary)
        validate_file(temporary, checkpoint.label)
        if sha256_digest(temporary) != downloaded_digest:
            raise RuntimeError("{} checksum validation failed.".format(checkpoint.label))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print("{}: downloaded and validated ({:,} bytes): {}".format(checkpoint.label, destination.stat().st_size, destination))


def main() -> int:
    repository = os.getenv("HF_MODEL_REPO", "").strip()
    if not repository:
        print("HF_MODEL_REPO is required, for example: your-account/geovision-checkpoints", file=sys.stderr)
        return 2

    token = os.getenv("HF_TOKEN", "").strip()
    try:
        for checkpoint in CHECKPOINTS:
            download_checkpoint(checkpoint, repository, token)
    except (RuntimeError, ValueError) as error:
        print("Checkpoint setup failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
