# Offline Demo Checklist

## Core local functionality

- [ ] Create an isolated Python environment and install `requirements-backend.txt`.
- [ ] Install frontend dependencies from the checked-in lockfile with `npm ci`.
- [ ] Run `python3 scripts/verify_installation.py`.
- [ ] Run `python3 scripts/verify_models.py`; do not replace a hash-mismatched checkpoint.
- [ ] Set `SATQUERY_GROUNDER_LOCAL_FILES_ONLY=true` after Grounding DINO/OpenCLIP/caption assets are cached.
- [ ] Start FastAPI on port 8010 and confirm `/api/agent/health` reports each specialist as ready, unloaded, disabled, or failed—never silently absent.
- [ ] Start Next.js and confirm its configured API base URL points to the local backend.
- [ ] Run `python3 scripts/smoke_all_workflows.py` and review every warning/fallback.
- [ ] Exercise Single Image, Optical+SAR, and Bi-temporal workflows from the UI.
- [ ] Download a JSON or PDF report and verify evidence provenance and limitations are present.
- [ ] Open SIH Compliance and confirm missing benchmarks say “Evidence gap,” not zero.

## Optional network-dependent functionality

- First-time Hugging Face acquisition is required only when the OpenCLIP, captioner, or Grounding DINO base assets are not already cached.
- Gemini is optional and is never required for core routing, evidence, grounding, VQA, change, fusion, or reports.
- TTP is an optional CUDA service. Local ChangerEx remains the default learned change engine; deterministic change remains the visible fallback.

## Scientific checks before judging

- Do not describe a PNG/JPEG pair as georeferenced. State that pixel alignment is supplied by the benchmark pairing and cannot be independently verified.
- Do not accept unknown multispectral RGB ordering. Confirm `selected_visual_bands` and `band_selection_reason`.
- Do not treat a translated SAR product as optical truth.
- Do not treat SVE similarity or model confidence as a calibrated probability.
- Do not claim RISAT or Cartosat-2S validation without actual approved samples.
