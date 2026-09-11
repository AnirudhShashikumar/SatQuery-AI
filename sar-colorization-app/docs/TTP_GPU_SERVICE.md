# TTP persistent GPU service

The `ttp_service` package is independently deployable and must run outside GeoVision's Apple Silicon environment on Python 3.10 with a CUDA 12.1-compatible NVIDIA host. It uses PyTorch `2.1.2+cu121`, torchvision `0.16.2+cu121`, torchaudio `2.1.2+cu121`, NumPy `1.26.4`, OpenCV headless `4.8.1.78`, MMCV `2.1.0`, MMEngine `0.10.7`, PEFT `0.8.2`, TorchMetrics `1.3.1`, Transformers `4.38.1`, and the other exact versions in `ttp_service/requirements-lock.txt`.

Bootstrap is fail-closed. It clones the official repository, checks out `431377d`, downloads `epoch_260.pth` from `KyanChen/TTP`, verifies its expected size range and exact SHA-256, and skips all network work when existing assets verify. The repository's bundled `mmseg` and `opencd` directories are placed ahead of installed packages. External OpenCD or MMSegmentation installations are unsupported.

The lifecycle is `unloaded → loading → ready` or `failed`. One `OpenCDInferencer` is constructed, retained in GPU memory, and serialized behind an inference lock. Health exposes counts and safe provenance only. Prediction measures load, inference, total request runtime, CUDA allocated/reserved/peak memory, and reuse state. CUDA OOM is caught, the CUDA cache is cleared best-effort, and the process remains alive.

Endpoints:

- `GET /health` — safe lifecycle, verification, model, device, counts, and limitations.
- `POST /predict` — multipart `earlier_image`, `later_image`, optional safe `request_id`.
- `GET /artifacts/{opaque-id}` — expiring PNG evidence with no filesystem path.

Run from the application root on the GPU host:

```bash
cd ttp_service
chmod +x setup_environment.sh
./setup_environment.sh
cd ..
MPLBACKEND=Agg PYTHONPATH="$PWD:/opt/ttp/TTP:/opt/ttp/TTP/mmseg:/opt/ttp/TTP/opencd" \
  ttp_service/.venv/bin/uvicorn ttp_service.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Docker:

```bash
docker build -t geovision-ttp:431377d ttp_service
docker run --rm --gpus all -p 8000:8000 --name geovision-ttp geovision-ttp:431377d
```

Use one Uvicorn worker so one checkpoint occupies GPU memory. Place TLS/authentication and upload-rate controls at the private deployment boundary. Do not expose the service publicly.
