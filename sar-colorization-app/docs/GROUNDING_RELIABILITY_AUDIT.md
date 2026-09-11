# Grounding reliability audit

Date: 2026-09-06

## Root cause

The production loader pointed `transformers` at `IDEA-Research/grounding-dino-tiny` with a temporary cache directory and defaulted `local_files_only` to false. During an offline start, Transformers therefore issued Hugging Face metadata requests and retried DNS failures even though a weight blob was present. The old temporary snapshot was also incomplete: `model.safetensors` existed, while the config, processor, and tokenizer symlinks referenced missing blobs. Health reported the configured/lazy object as available without proving that processor inference, model inference, specialist rescoring, and post-processing worked.

## Repair

- Added a durable repository-local official snapshot at `models/grounding_dino_tiny`.
- Local resolution now checks an explicit `SATQUERY_GROUNDER_LOCAL_MODEL`, a repository bundle, standard Hugging Face locations, `HF_HOME`, `TRANSFORMERS_CACHE`, and the legacy SatQuery temporary cache, in that order.
- Required config, processor, tokenizer, vocabulary, and safetensors files are verified before loading.
- Demo-safe startup is offline by default. A network download requires explicit `SATQUERY_GROUNDER_ALLOW_DOWNLOAD=true`; no runtime download is silent.
- Startup runs a generated 32×32 smoke through the real processor, Grounding DINO model, Grounding Specialist v1.1 proposal head, and Hugging Face post-processor. `ready` is published only after that path succeeds.
- Health now exposes `status`, selected `device`, `model_id`, `load_source`, `last_error`, and `smoke_verified`.
- Missing files, invalid paths, processor failures, checkpoint incompatibility, MPS initialization errors, and generic initialization exceptions have separate safe public reasons. Raw tracebacks remain server-side.
- MPS remains preferred when PyTorch reports it available. Existing inference-time fallback moves the same loaded model to CPU and discloses the fallback.
- Grounding Specialist v1.1 remains in the inference path. If its checkpoint or rescoring fails, base Grounding DINO is only a disclosed degraded fallback; the specialist is not reported ready.

## Verified local runtime

The official snapshot resolved from `models/grounding_dino_tiny`, loaded with `local_files_only=True`, and completed the real smoke on CPU. A separate isolated application startup selected MPS and reported both Grounding DINO and Grounding Specialist v1.1 ready after smoke verification. The five representative queries all executed against one reused model instance on `satquery_agent/demo_samples/single-optical.png`:

| Query | Canonical target | Accepted regions | Overlay |
|---|---:|---:|---:|
| Highlight the water body. | water body | 2 | yes |
| Locate the built-up area. | building | 1 | yes |
| Where are the buildings? | building | 1 | yes |
| Highlight the river. | river | 2 | yes |
| Find the large water-covered region. | water body | 2 | yes |

These counts are smoke evidence for this image, not benchmark results. Detection scores remain uncalibrated alignment/proposal scores.

The manual browser workflow also completed `Locate the buildings.` with one accepted region, a grounding overlay, explicit Grounding DINO + Grounding Specialist v1.1 provenance, operational confidence, and scientific limitations.

## Integrity

- Grounding DINO architecture: unchanged.
- Grounding Specialist v1.1 head: unchanged; SHA-256 remains `5e8db30becadb1d063fc0154614ee2a2a4b7a2c2923db3fce4ff036c65007342`.
- VRSBench validation: unchanged at 16,146 samples, mean IoU 0.2348, Acc@0.50 25.05%.
- No generic VLM fallback or hardcoded detection answer was added.
