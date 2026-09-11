# SAR-to-Optical Single-Image Integration Plan

## Decision

Use a shared in-process service at `satquery_agent/services/sar_translation_service.py`. It owns translation configuration, eligibility, lifecycle, locking, preprocessing, inference, fallback, safe metadata, and artifacts. When `backend.py` already owns loaded model objects, it registers those exact objects with the singleton. Otherwise the service lazily loads at most one instance of each required model.

The existing router, endpoints, native specialists, optical behavior, checkpoint contents, and response fields remain unchanged. The entire additional branch is disabled unless `SATQUERY_SAR_TRANSLATION_ENABLED=1`.

## Options considered

| Option | Circular imports | Model memory | Latency | Lifecycle/error ownership | Testability/deployment |
|---|---|---|---|---|---|
| A. Import `backend.py` from the agent | Unsafe: backend imports agent API | Reuses globals only after a circular import succeeds | Low after import, but import has eager model cost | Split and fragile | Poor; importing agent starts backend models/services |
| B. Shared service with registration | None; backend calls down into agent service | One instance per model in the host process | In-process, no serialization | One explicit translation lifecycle and safe error surface | Strong; factories/models are injectable |
| C. Internal HTTP call to reconstruction endpoints | None | One backend process may reuse models; extra process can duplicate | Highest; encode/upload/decode overhead | Network and HTTP errors added | Requires a listening backend and complicates unit tests |
| D. Independent agent model instances | None | Duplicates GPU/CPU memory and checkpoint loading | Low after duplicate load | Competing lifecycle state | Simple locally but violates reuse requirement |

Option B is selected because it is the only approach that is circular-import safe, in-process, testable, and capable of reusing the models already loaded by the current host.

## Configuration

- `SATQUERY_SAR_TRANSLATION_ENABLED`: default `false`; the sole master switch.
- `SATQUERY_SAR_TRANSLATION_MODEL`: `sarfusionformer` (default) or `pix2pix`; invalid values are reported safely.
- `SATQUERY_SAR_TRANSLATION_USE_COLOR_CORRECTION`: when absent, enabled only if a corrector checkpoint/registered corrector is available.
- `SATQUERY_SAR_TRANSLATION_OPTICAL_SPECIALISTS_ENABLED`: default `true`; controls only downstream optical evidence.
- `SATQUERY_SAR_TRANSLATION_SAVE_ARTIFACTS`: default `true`; uses the secure preview store.

These values are included in the agent cache identity when the primary effective modality is SAR.

## Internal interfaces

`SarTranslationService.evaluate_eligibility(raster, metadata, model_image)` returns a structured decision with eligibility, requested/selected model, fallback eligibility, channel interpretation, finite ratio, dimensions, dtype, preprocessing, and a safe reason.

`SarTranslationService.translate(...)` returns an internal result containing an owned PIL RGB image and JSON-safe metadata: dimensions, model, fallback/color-correction flags, device, runtimes, preprocessing, channel interpretation, output range, warnings, provenance, artifact URLs, and SHA-256 content hash. It never returns tensors through a response model.

`register_external_models(...)` accepts backend-owned model instances and their safe load errors. Identity checks prevent replacement of an already-ready different instance. `load()` is double-checked under a re-entrant load lock; inference is serialized under a separate lock. Model state is `disabled`, `unloaded`, `loading`, `ready`, or `failed`. Failure is sticky until explicit `retry()`.

## Eligibility and preprocessing

- SARFusionFormer requires a scientific, verified `SAR_VV_VH` input with exactly two raster channels. Channel 0 is VV and channel 1 is VH. Both must have finite values. Each channel uses its finite 1st/99th percentiles, invalid values are replaced only after deriving those percentiles, and the pair is resized to `[1,2,256,256]`.
- Pix2Pix accepts an existing SAR display representation. The ingestion-owned RGB preview is resized to `256×256` and normalized to `[-1,1]`. For a one-band preview, the explicit conversion is “normalized grayscale display replicated to RGB by PIL”; it is recorded in metadata and never described as a VV/VH pair.
- Unsupported/ambiguous inputs skip translation or use only the compatible fallback. No channel is silently duplicated to satisfy SARFusionFormer.

Primary selection is configurable. If the requested model is unavailable or ineligible, the other model may be used only when its own eligibility contract passes. `fallback_used` and the reason are always disclosed.

## Generated-image contract

Downstream specialists receive a copied RGB PIL image and synthetic `ImageMetadata` that describes the generated display, uses its true generated width/height and three bands, has no CRS/transform/bounds, and is never presented as observed optical data. Grounding boxes remain coordinates on this generated representation. Source-coordinate mapping is not performed because translation resizes and does not establish a verified invertible spatial transform.

## Native and translated branches

The current routed native specialist remains authoritative. When translation is enabled on a SAR single image, a native scene summary is also collected if the routed task did not already produce native SAR evidence. Failure in this supporting summary does not erase the routed result.

The translated branch invokes only specialists justified by the existing query classifiers:

- SVE for scene priors when enabled;
- captioner for descriptive/SAR-scene requests;
- grounder only when the existing grounding classifier recognizes a valid supported target;
- RSVQA only when `classify_rsvqa_task` recognizes an exported family.

Each failure is isolated and produces a bounded warning. No optional failure crashes the request.

## Deterministic fusion

`satquery_agent/specialists/sar_optical_fusion.py` consumes native result models, translated evidence, query, and modality metadata. It does not call an LLM. Water agreement uses native water-candidate evidence and controlled water terms in translated outputs. Quality failure lowers confidence. Other generated semantic observations remain `suggested_by_translation_only` unless a conservative native rule independently supports them.

States are `supported_by_both`, `supported_by_native_sar_only`, `suggested_by_translation_only`, `conflicting_evidence`, `insufficient_evidence`, or `translation_unavailable`. Required disclosures and evidence-source wording are assembled deterministically.

## Optional response model

Add `AgentResponse.sar_translated_optical_analysis: Optional[SarTranslatedOpticalAnalysis] = None`. The nested JSON-safe model contains status/configuration, generated preview, disclosures, optical evidence, agreement, confidence, native/translated findings, warnings, limitations, provenance, and runtime breakdown. Existing fields are neither renamed nor made required.

## Artifacts and ownership

When enabled by configuration, save the normalized SAR preview, generated optical-like image, corrected image when distinct, grounding overlay when produced, and existing native water artifacts through `image_ingestion.save_preview`. Filenames remain random UUIDs in the configured temporary directory. The service returns one owned generated PIL object to orchestration; orchestration closes it in `finally` after all optical specialists complete.

## Health, registry, and observability

`/api/agent/health` exposes `sar_translation_service` using optional health fields for enabled state, selected model, fallback availability, color-corrector availability, load/reuse counts, and safe error. The root `/health` includes the same bounded status without changing existing model entries. The registry describes generated optical-like imagery as secondary interpretive evidence, never observed optical measurement.

Execution uses the required bounded stage IDs: eligibility, preprocessing, model load/reuse, inference, correction, artifacts, individual translated optical specialists, and fusion. Parameters contain only statuses, model identifiers, counts, booleans, dimensions, and safe reasons—never pixels, tensors, filenames, or full prompts.

## Failure policy

Translation disabled returns no new response field and executes no new code path. Unsupported input returns an optional `translation_unavailable` result while retaining native output. Missing checkpoints, load/inference/correction/artifact/specialist failures are caught at their boundary, converted to safe warnings, and do not expose tracebacks. A primary translator failure attempts the compatible fallback; a downstream specialist failure does not stop other evidence.

## Verification plan

Focused tests cover flags, eligibility, preprocessing/output contracts, primary/fallback/correction behavior, lazy reuse and locking, sticky failure/retry, native preservation, selective optical calls, deterministic fusion states, schemas, health, and default-off compatibility. A smoke script exercises native-only, enabled, and forced-fallback paths using an approved local SAR sample, then writes bounded results and runtime artifacts under `artifacts/single_sar_translation_smoke/`.

