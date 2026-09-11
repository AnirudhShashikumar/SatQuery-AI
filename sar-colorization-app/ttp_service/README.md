# GeoVision TTP CUDA service

This independently deployable Python 3.10 service keeps the official Time Travelling Pixels model resident on one CUDA worker. It uses the official `KyanChen/TTP` repository pinned to commit `431377d`, the Hugging Face `KyanChen/TTP` checkpoint `epoch_260.pth`, and requires SHA-256 `60294429b3d22310e1451b1059b953b44323adfe3af4eae1d75ae1709e610cb9` before model loading.

The repository's bundled `mmseg` and `opencd` directories are placed ahead of site packages. Do not install external `mmsegmentation` or OpenCD packages. The upstream repository declares an Apache-2.0 license; retain its license and notices with deployments.

## Run on a CUDA 12.1 host

From the application root:

```bash
cd ttp_service
./setup_environment.sh
cd ..
MPLBACKEND=Agg PYTHONPATH="$PWD:/opt/ttp/TTP:/opt/ttp/TTP/mmseg:/opt/ttp/TTP/opencd" \
  ttp_service/.venv/bin/uvicorn ttp_service.app:app --host 0.0.0.0 --port 8000 --workers 1
```

`GET /health` reports only safe lifecycle/provenance details. `POST /predict` accepts `earlier_image`, `later_image`, and an optional opaque `request_id`. Binary masks and display products are returned as expiring opaque `/artifacts/{id}` references. No local paths are returned.

Set `TTP_LOAD_ON_STARTUP=true` (the default) so model download/verification and loading happen before a presentation. A failed verification prevents loading. Recoverable prediction failures do not terminate the process; CUDA OOM triggers a best-effort cache cleanup.

TTP produces a model-generated binary change prediction, not ground truth. It was trained on LEVIR-CD building-change imagery and does not infer semantic change type or cause.
