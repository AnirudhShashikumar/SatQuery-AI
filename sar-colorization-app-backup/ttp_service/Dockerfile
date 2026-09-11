FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
    TTP_REPOSITORY_DIR=/opt/ttp/TTP TTP_ASSET_DIR=/opt/ttp/assets TTP_CHECKPOINT_PATH=/opt/ttp/assets/epoch_260.pth \
    PYTHONPATH=/app:/opt/ttp/TTP:/opt/ttp/TTP/mmseg:/opt/ttp/TTP/opencd
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3-pip git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-lock.txt /app/requirements-lock.txt
RUN python3.10 -m pip install --no-cache-dir --upgrade pip wheel setuptools && \
    python3.10 -m pip install --no-cache-dir torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    python3.10 -m pip install --no-cache-dir openmim && mim install mmcv==2.1.0 && \
    python3.10 -m pip install --no-cache-dir -r /app/requirements-lock.txt
COPY . /app/ttp_service
RUN python3.10 -m ttp_service.download_assets --repository /opt/ttp/TTP --assets /opt/ttp/assets && \
    python3.10 -m ttp_service.verify_assets --repository /opt/ttp/TTP --checkpoint /opt/ttp/assets/epoch_260.pth
EXPOSE 8000
CMD ["python3.10", "-m", "uvicorn", "ttp_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
