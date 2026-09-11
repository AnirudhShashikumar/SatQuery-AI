# SatQuery AI

SatQuery AI is a local, auditable remote-sensing assistant for ISRO SIH Problem Statement 26167. It validates uploaded imagery, routes the request to a compatible specialist, executes the real local workflow, and returns evidence, provenance, confidence disclosures, and an observable trace.

Supported SatQuery workflows:

- optical/RGB-like multispectral scene captioning with the RSICD-adapted BLIP specialist;
- controlled single-image VQA over deterministic visible-spectrum evidence;
- local zero-shot text-guided box grounding for optical/RGB-like multispectral imagery with the official Grounding DINO tiny checkpoint;
- ChangerEx-first bi-temporal analysis with independent deterministic evidence, evidence-gated semantic answers, and visible deterministic fallback;
- exactly aligned optical and native-SAR evidence fusion with source-labelled facts and structured questions;
- optional, default-off single-image SAR-to-optical supporting evidence with deterministic native/generated fusion;
- backend-authoritative PDF, JSON, CSV, and ZIP mission reports.

Direct general SAR grounding/captioning and single-image SAR VQA remain explicitly unavailable. When `SATQUERY_SAR_TRANSLATION_ENABLED=1`, compatible optical specialists may interpret a generated optical-like representation as secondary evidence with prominent provenance and limitations; it is never described as observed optical data. Grounding boxes are model-produced candidates, not ground truth, and SAM/SAM2 mask refinement is not connected. No hosted VLM is required by SatQuery’s grounding, deterministic VQA, change, cross-modal, or reporting workflows. See [Single-Image SAR Translation Evidence](docs/SINGLE_IMAGE_SAR_TRANSLATION.md).

The repository also retains the two original reconstruction workflows below without changing their model behavior.

A Streamlit interface and FastAPI service for two independent SAR-to-optical workflows:

- **Pix2Pix** for visually realistic optical reconstruction from the repository's existing SAR image representation.
- **SARFusionFormer** for structure-preserving reconstruction from separate raw VV and VH channels.

The application compares the two results side by side but never blends them.

## Checkpoint placement

Keep model weights out of Git. The default local paths are:

```text
../pix2pix_gen_180.pth
../models/checkpoints/sarfusionformer_256_decoder_best.pt
../models/checkpoints/color_corrector_256_best.pt
```

The color-corrector is optional and is disabled by default. Verify the raw SARFusionFormer result first, then explicitly enable colour correction if desired. If its checkpoint is unavailable, raw SARFusionFormer inference remains available.

Required SatQuery specialist bundles live under `models/`. Verify all known hashes without loading weights:

```bash
python3 scripts/verify_models.py
```

## Clean installation

Use an isolated environment. Python 3.11 or 3.12 and Node.js 22 LTS are recommended; the historical local Python 3.9 environment is still exercised but emits upstream end-of-life warnings.

```bash
python3.11 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements-backend.txt
cd frontend
npm ci
cd ..
python3 scripts/verify_installation.py
```

No setup command modifies model weights. First use of OpenCLIP, Grounding DINO, or captioning may require network access unless the upstream assets are already in `SATQUERY_MODEL_CACHE`.

## Run locally

From this directory, start the API:

```bash
venv/bin/uvicorn backend:app --host 127.0.0.1 --port 8010
```

In a second terminal, start the interface:

```bash
venv/bin/streamlit run app.py
```

For the presentation-ready Next.js interface, use:

```bash
cd frontend
npm run dev
```

Then open `http://127.0.0.1:3000/assistant`.

Recent authoritative results from Assistant, direct change/cross-modal analysis, Pix2Pix, and SARFusionFormer appear at `http://127.0.0.1:3000/assistant/compare`. The existing `/comparison` route remains the focused reconstruction-only workspace.

The interface connects to `http://127.0.0.1:8010` by default.

## Environment variables

- `PIX2PIX_CHECKPOINT`: path to the Pix2Pix generator checkpoint.
- `SARFUSIONFORMER_CHECKPOINT`: path to `sarfusionformer_256_decoder_best.pt`.
- `COLOR_CORRECTOR_CHECKPOINT`: path to `color_corrector_256_best.pt`.
- `SAR_COLORIZATION_API_URL`: API URL used by Streamlit.
- `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_DEFAULT_MODEL`, `GEMINI_SUPPORTED_MODELS`: optional server-side Gemini configuration. On local macOS installs, configure a Gemini key through Settings → AI Providers; it is stored in macOS Keychain and never added to Git. Do not set `NEXT_PUBLIC_GEMINI_API_KEY`.
- `GEOVISION_SECRET_ENCRYPTION_KEY`: reserved for an encrypted server-side secret-store deployment; keep it server-only and out of Git.
- `CORS_ORIGINS`: comma-separated trusted frontend origins; defaults to the local GeoVision frontend.
- `SATQUERY_DEMO_MODE=true`: enables only the checked-in local demo manifest and sample-file endpoints. It is disabled by default.
- `SATQUERY_RESULT_CACHE_MAX_ITEMS`: maximum process-local SatQuery result records; default `32`.
- `SATQUERY_RESULT_CACHE_TTL_SECONDS`: result/cache expiry; default `1800` seconds.
- `SATQUERY_COMPARISON_MAX_ITEMS`: maximum process-local normalized comparison records; default `32`.
- `SATQUERY_COMPARISON_TTL_SECONDS`: comparison history expiry; default `1800` seconds.
- `SATQUERY_REPORT_MAX_ARTIFACTS`: maximum temporary report artifacts; default `128`.
- `SATQUERY_REPORT_TTL_SECONDS`: report artifact expiry; default `1800` seconds.
- `SATQUERY_MAX_UPLOAD_MB`: per-image upload limit; default `100` MB.
- `SATQUERY_GROUNDER_ENABLED`: enables the local Grounding DINO specialist; default `true`.
- `SATQUERY_GROUNDER_CHECKPOINT`: local/Hugging Face checkpoint identifier; default `IDEA-Research/grounding-dino-tiny`.
- `SATQUERY_GROUNDER_LOCAL_FILES_ONLY=true`: disables checkpoint network access after the model is cached.
- `SATQUERY_GROUNDING_SPECIALIST_SCORE_MODE`: Grounding Specialist v1.1 score policy: `disabled`, `fallback`, `blend`, or `specialist_only`; default `fallback`. Fallback mode preserves the Grounding DINO accepted boxes and labels and reverts automatically for missing, invalid, non-finite, out-of-range, or extremely small specialist probabilities.
- `SATQUERY_GROUNDING_DINO_WEIGHT`, `SATQUERY_GROUNDING_SPECIALIST_WEIGHT`: normalized blend weights used only in `blend` mode; defaults `0.70` and `0.30`. Specialist sigmoid probabilities are min-max normalized across accepted proposals before blending; these scores are still operational rankings rather than calibrated real-world probabilities.
- `SATQUERY_CHANGE_ENGINE`: selects `changerex` (default local learned engine), `ttp` (optional alternate service), or `deterministic`.
- `SATQUERY_CHANGEREX_CHECKPOINT`: official SHA-256-verified ChangerEx ResNet-18 checkpoint. ChangerEx is loaded lazily once on MPS or CPU and its learned mask is primary; the deterministic mask remains independently labelled support and the exact fallback.
- `SATQUERY_CHANGEREX_DEVICE`: `auto`, `mps`, or `cpu`. An unavailable or failed model never crashes the request and produces a visible deterministic-fallback warning.
- `TTP_ENABLED`: enables the optional TTP alternate engine for eligible aligned optical pairs; default `true`.
- `TTP_DEFAULT_MODE`: legacy compatibility setting for deployments that have not set `SATQUERY_CHANGE_ENGINE`. Explicit legacy `hybrid` or `ttp` values continue to select TTP.
- `TTP_SERVICE_URL`: private URL of the single-worker CUDA service holding the verified `epoch_260.pth` checkpoint. Unavailable, invalid, or failed learned inference automatically uses the deterministic pipeline.
- `TTP_LOAD_ON_STARTUP`: CUDA-service option; default `false` for first-use lazy loading. One loaded model is retained and serialized for all subsequent requests.
- `SATQUERY_MODEL_CACHE`: shared local model-cache directory for captioning and grounding.

## Vercel frontend deployment

The Next.js dashboard is ready to deploy as a Vercel project. This repository also
contains a PyTorch inference API and model checkpoints; keep that API on a separate
long-running compute service rather than deploying it as a Vercel Function.

When importing this repository into Vercel, configure the project as follows:

- **Root Directory:** `sar-colorization-app/frontend`
- **Framework Preset:** Next.js (the detected default)
- **Build Command / Output Directory:** leave at the detected Next.js defaults
- **Environment Variable:** set `NEXT_PUBLIC_API_URL` to the public HTTPS URL of the
  deployed inference API, with no trailing slash

On the API host, set `CORS_ORIGINS` to the Vercel production domain (and any preview
domains that should call the API), and configure the three checkpoint paths. Keep
Gemini and encryption secrets server-side; never add them as `NEXT_PUBLIC_*` values.

## API routes

- `POST /predict`: legacy Pix2Pix PNG response.
- `POST /evaluate`: legacy Pix2Pix PNG response with PSNR and SSIM headers.
- `POST /api/pix2pix/infer`: Pix2Pix JSON inference response.
- `POST /api/sarfusionformer/infer`: independent VV/VH SARFusionFormer inference response.
- `POST /api/compare`: metrics for two already-generated outputs and one common ground truth.
- `POST /api/analysis/image`: optional qualitative analysis of a rendered image; it never changes inference or metrics.
- `POST /api/agent/change`: ChangerEx-first multipart bi-temporal analysis for `before_image`, `after_image`, `before_date`, and `after_date`; returns learned/deterministic evidence, regions, statistics, compatibility, local semantic interpretation, warnings, and an execution trace. Unsupported semantic causes remain blocked.
- `POST /api/agent/cross-modal`: deterministic multipart optical–SAR evidence fusion for `optical_image` and `sar_image`; full pixel analysis requires exact geospatial alignment and returns water-likelihood, structural-likelihood, visible-spectrum vegetation support, agreement/disagreement, regions, previews, confidence rationale, and a safe execution trace.
- `POST /api/agent/query`: includes real local Grounding DINO box grounding, controlled local VQA for supported single-image optical questions, measured bi-temporal change questions, and optical–SAR evidence questions. No generic LLM or hosted VLM is used for these workflows.
- `POST /api/agent/report`: generates temporary backend-authoritative mission artifacts for a stored `request_id` in PDF, JSON, CSV, or ZIP form.
- `GET /api/agent/comparison-items`: returns bounded, preview-safe summaries of recent comparable workflow results.
- `GET /api/agent/comparison-items/{request_id}`: returns the complete authoritative normalized comparison item or an explicit missing/expired response.
- `POST /api/agent/comparison-assessment`: evaluates two to four stored IDs with transparent `direct`, `partial`, or `not_direct` rules.
- `POST /api/agent/comparison-report`: generates PDF, JSON, and ZIP comparison artifacts from stored backend values; optional user notes remain explicitly labelled.
- `GET /api/agent/reports/{artifact}`: downloads a safe temporary report artifact.
- `GET /api/agent/compliance`: truthful SIH requirement matrix used by `/assistant/compliance`.
- `GET /api/agent/tools`: public model/tool registry, including unavailable specialists.
- `GET /api/agent/demo`: environment-gated local approved-sample manifest.

Controlled VQA validation can be run with `venv/bin/python scripts/evaluate_controlled_vqa.py manifest.json --output predictions.jsonl`. This is a project validation utility, not an RSVQA benchmark claim.

## Tests and evaluation

```bash
# Backend
venv/bin/python -m pytest -q
venv/bin/python scripts/validate_benchmark_results.py
venv/bin/python scripts/smoke_all_workflows.py --output artifacts/final_readiness/workflow_smoke.json

# Frontend
cd frontend
npm test
npx tsc --noEmit --incremental false
npm run lint
npm run build
```

Public-dataset runners do not download data and fail clearly when labels/splits are absent:

```bash
python3 scripts/evaluate_cdvqa.py --dataset-root /path/to/CDVQA --split test --device auto --output-dir artifacts/cdvqa_test
python3 scripts/evaluate_changerex_levircd.py --dataset-root /path/to/LEVIR-CD --split test --device auto --output-dir artifacts/changerex_levircd_test
python3 scripts/evaluate_optical_sar.py --manifest /path/to/controlled.jsonl --dataset-root /path/to/images
python3 scripts/evaluate_sar_translation.py --manifest /path/to/paired.jsonl --dataset-root /path/to/images
python3 scripts/evaluate_captions.py --manifest /path/to/captions.jsonl --dataset-root /path/to/images
python3 scripts/calibrate_confidence.py --input /path/to/confidence_results.csv
python3 scripts/benchmark_performance.py --output-dir artifacts/performance
```

Evaluator outputs retain model/dataset/split/checkpoint provenance. Smoke, parity, latency, and learned-versus-deterministic overlap are never reported as accuracy.

- `GET|POST|DELETE /api/settings/ai-provider`: retrieve non-sensitive Gemini status, validate and save a local secure configuration, or remove it. Legacy `/api/settings/provider` routes remain available.
- `GET /api/settings/ai-provider/models?provider=gemini`: Gemini image-capable models enabled by this backend.
- `POST /api/settings/ai-provider/test`: lightweight server-side Gemini connection test. Legacy `/api/settings/test` remains available.
- `GET /health`: independent availability for all three loaded model components.

## Input requirements

Pix2Pix accepts the same RGB image formats as the original project. SARFusionFormer defaults to one combined `.npy` input containing VV and VH as `[2, H, W]`, `[H, W, 2]`, `[1, 2, H, W]`, or `[1, H, W, 2]`. It also accepts separate VV and VH files as `.npy`, TIFF, PNG, or JPEG; those channels must have matching spatial dimensions.

SARFusionFormer replaces invalid numeric values with zero, independently percentile-normalizes VV and VH using the 1st and 99th percentiles, and resizes both channels to 256 × 256 before inference. The backend uses the training model's GroupNorm/GELU, shifted-window attention, decoder, and normalized-Lab-to-sRGB conversion exactly; it does not apply log/dB, mean/std, gamma, histogram, global, or `[-1,1]` input normalization.

## Scientific output and display output

SARFusionFormer returns both a raw radiometric PNG and an enhanced display PNG. The latter applies one global 2nd/98th-percentile stretch across the complete RGB image solely for on-screen inspection; it never changes channel balance. Metrics, raw downloads, and the optional colour corrector always use the unmodified float RGB model output.

## Mission reports and cache behavior

Successful and partial Assistant results expose PDF, JSON, and Full ZIP actions. Reports are generated from the authoritative backend response, not frontend-provided measurements. ZIP packages include the PDF, typed JSON, numeric/region CSV, evidence PNGs, and a content README. They never include original uploads, API keys, environment values, model-cache paths, hidden reasoning, or source filesystem paths.

Result and artifact storage is bounded, temporary, process-local, and cleared by backend restart. Cache identity includes input SHA-256 hashes, normalized query, routed task, modalities, dates/safe parameters, and the SatQuery tool version. Cached results disclose original and retrieval timestamps; forced reruns bypass cache reuse.

Mission Comparison uses SHA-256 content identity rather than filenames. It preserves task-specific statistics, exposes non-comparability instead of inventing shared scores, supports derived-output lineage when a stored output hash becomes a later input, and never declares a universal winner.

## Offline and demo behavior

All deterministic workflows, reports, compliance data, and approved samples operate locally. Captioning and Grounding DINO are offline after their Hugging Face checkpoints are cached; use `SATQUERY_GROUNDER_LOCAL_FILES_ONLY=true` to enforce offline grounding. Demo mode never downloads samples or hardcodes answers; it loads checked-in fixtures and executes the same `/api/agent/query` pipeline.

See:

- [SatQuery architecture](docs/SATQUERY_ARCHITECTURE.md)
- [SIH compliance matrix](docs/SATQUERY_COMPLIANCE.md)
- [Offline demo runbook](docs/SATQUERY_DEMO.md)
- [Scientific limitations](docs/SATQUERY_LIMITATIONS.md)
- [Problem Statement 26167 compliance](docs/PROBLEM_STATEMENT_26167_COMPLIANCE.md)
- [Model and dataset provenance](docs/MODEL_DATASET_PROVENANCE.md)
- [Cartosat/RISAT readiness](docs/CARTOSAT_RISAT_READINESS.md)
- [Offline demonstration checklist](docs/OFFLINE_DEMO_CHECKLIST.md)
- [Final submission audit](docs/SIH_FINAL_SUBMISSION_AUDIT.md)

## Verification snapshot

- 573 backend tests pass; five environment/checkpoint-dependent tests are skipped in the standard suite.
- 193 frontend tests across 33 files pass.
- Python compilation, TypeScript checking, ESLint with zero errors (27 documented `no-img-element` warnings), and the Next.js production build pass.
- Assistant, Mission Comparison, Research Analytics, SIH Compliance, architecture, Pix2Pix, SARFusionFormer, Model Comparison, benchmark, reports, and settings routes remain protected by regression and route checks.

These are project verification results, not public benchmark claims and not evidence of operational ISRO deployment.

Optical + SAR Joint Analysis validates the Optical and SAR upload slots against detected content. Reversed files must be explicitly swapped; ambiguous files need replacement. Same-size PNG/JPEG demo pairs can answer qualitative questions with native evidence from both sources. Geospatial co-registration remains unverified, so joint percentages, intersection masks, coordinates and real-world areas remain unavailable. See [the pair protocol](docs/OPTICAL_SAR_EVALUATION_PROTOCOL.md).
