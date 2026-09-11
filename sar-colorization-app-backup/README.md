# GeoVision · SatQuery AI

GeoVision is a local, auditable remote-sensing assistant for ISRO SIH Problem Statement 26167. SatQuery validates uploaded imagery, routes the request to a compatible specialist, executes the real local workflow, and returns evidence, provenance, confidence disclosures, and an observable trace.

Supported SatQuery workflows:

- optical/RGB-like multispectral scene captioning with the RSICD-adapted BLIP specialist;
- controlled single-image VQA over deterministic visible-spectrum evidence;
- local zero-shot text-guided box grounding for optical/RGB-like multispectral imagery with the official Grounding DINO tiny checkpoint;
- deterministic bi-temporal change analysis and controlled change questions;
- deterministic, exactly aligned optical–SAR evidence fusion and structured questions;
- backend-authoritative PDF, JSON, CSV, and ZIP mission reports.

General SAR grounding/captioning and single-image SAR VQA remain explicitly unavailable. Grounding boxes are model-produced candidates, not ground truth, and SAM/SAM2 mask refinement is not connected. No hosted VLM is required by SatQuery’s grounding, deterministic VQA, change, cross-modal, or reporting workflows.

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
- `POST /api/agent/change`: deterministic multipart bi-temporal analysis for `before_image`, `after_image`, `before_date`, and `after_date`; returns visual difference products, connected regions, statistics, compatibility, warnings, and an execution trace without AI-generated interpretation.
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

## Verification snapshot

- 179 backend tests pass; three opt-in real-checkpoint tests are skipped in the standard suite (captioner and two grounding integration checks).
- 32 frontend tests pass.
- Python compilation, TypeScript checking, and the Next.js production build pass.
- Assistant, Mission Comparison, Research Analytics, SIH Compliance, architecture, Pix2Pix, SARFusionFormer, Model Comparison, benchmark, reports, and settings routes remain protected by regression and route checks.

These are project verification results, not public benchmark claims and not evidence of operational ISRO deployment.
