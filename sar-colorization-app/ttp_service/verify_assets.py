"""Fail-closed verification for the official pinned TTP assets."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from .schemas import CHECKPOINT_SHA256


MIN_CHECKPOINT_BYTES = 1_200_000_000
MAX_CHECKPOINT_BYTES = 1_350_000_000
EXPECTED_COMMIT = "431377d"


def verify_checkpoint(path: Path, expected_sha256: str = CHECKPOINT_SHA256) -> None:
    if not path.is_file():
        raise FileNotFoundError("The TTP checkpoint is missing.")
    size = path.stat().st_size
    if not MIN_CHECKPOINT_BYTES <= size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("The TTP checkpoint size is outside the verified range.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("The TTP checkpoint checksum does not match the pinned artifact.")


def verify_repository(path: Path) -> None:
    if not (path / ".git").is_dir() or not (path / "mmseg").is_dir() or not (path / "opencd").is_dir():
        raise ValueError("The bundled TTP repository is incomplete.")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
    if not commit.startswith(EXPECTED_COMMIT):
        raise ValueError("The TTP repository is not at the pinned commit.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    verify_repository(args.repository)
    verify_checkpoint(args.checkpoint)
    print("TTP repository and checkpoint verified.")


if __name__ == "__main__":
    main()
