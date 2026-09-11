#!/usr/bin/env bash
set -euo pipefail

python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install openmim
mim install mmcv==2.1.0
python -m pip install -r requirements-lock.txt
python -m ttp_service.download_assets --repository "${TTP_REPOSITORY_DIR:-/opt/ttp/TTP}" --assets "${TTP_ASSET_DIR:-/opt/ttp/assets}"
python -m ttp_service.verify_assets --repository "${TTP_REPOSITORY_DIR:-/opt/ttp/TTP}" --checkpoint "${TTP_CHECKPOINT_PATH:-/opt/ttp/assets/epoch_260.pth}"
