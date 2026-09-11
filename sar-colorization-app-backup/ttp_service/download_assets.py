"""Idempotently acquire the pinned official repository and Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

from .verify_assets import EXPECTED_COMMIT, verify_checkpoint, verify_repository


REPOSITORY_URL = "https://github.com/KyanChen/TTP.git"
HF_REPOSITORY = "KyanChen/TTP"
CHECKPOINT_FILENAME = "epoch_260.pth"


def acquire(repository: Path, assets: Path) -> None:
    repository.parent.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    if repository.exists():
        verify_repository(repository)
    else:
        subprocess.run(["git", "clone", "--filter=blob:none", REPOSITORY_URL, str(repository)], check=True)
        subprocess.run(["git", "checkout", EXPECTED_COMMIT], cwd=repository, check=True)
        verify_repository(repository)
    checkpoint = assets / CHECKPOINT_FILENAME
    if checkpoint.exists():
        verify_checkpoint(checkpoint)
        return
    downloaded = Path(hf_hub_download(repo_id=HF_REPOSITORY, filename=CHECKPOINT_FILENAME, local_dir=assets))
    verify_checkpoint(downloaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("/opt/ttp/TTP"))
    parser.add_argument("--assets", type=Path, default=Path("/opt/ttp/assets"))
    args = parser.parse_args()
    acquire(args.repository, args.assets)


if __name__ == "__main__":
    main()
