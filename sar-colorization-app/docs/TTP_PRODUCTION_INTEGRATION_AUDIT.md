# TTP production integration audit

## Scope and conclusion

SatQuery already contains most of a production-safe TTP integration, split between the main backend and an isolated CUDA service. The learned path is not active in the current production configuration because `TTP_ENABLED` defaults to `false`; the configured service at `127.0.0.1:8000` is not running; and neither the pinned TTP repository nor the verified `epoch_260.pth` checkpoint exists on this host. Successful TTP responses already become primary change evidence, but several lifecycle, health, lazy-loading, trace, and smoke-validation connections are incomplete.

This audit was completed before production code changes. Files inspected include `backend.py`, `satquery_agent/api.py`, `router.py`, `models.py`, `registry.py`, `specialists/ttp_change.py`, `specialists/change_analysis.py`, `scripts/benchmark_ttp.py`, the complete `ttp_service` package, existing TTP documentation, and all related tests.

## 1. Current TTP architecture

The implementation has two processes:

1. The SatQuery backend owns routing, ingestion, pair validation, deterministic change analysis, controlled VQA, reports, analytics, and a singleton `TTPServiceClient` named `TTP_CLIENT`.
2. The separately deployable `ttp_service` owns the official TTP repository, the verified checkpoint, CUDA inference, model lifecycle, mask artifacts, and GPU metrics.

The learned model is the official KyanChen/TTP implementation pinned to repository commit `431377d`. Provenance constants identify `epoch_260.pth`, SHA-256 `60294429b3d22310e1451b1059b953b44323adfe3af4eae1d75ae1709e610cb9`, architecture `SAM ViT-L + LoRA SiamEncoderDecoder`, and LEVIR-CD training data.

## 2. Current execution flow

For `/api/agent/change` and routed bi-temporal `/api/agent/query` requests, SatQuery:

1. Ingests both images.
2. Validates modality, dimensions, dates, alignment, CRS, and bounds.
3. Runs the deterministic normalized-difference analyzer.
4. Checks requested TTP mode and optical eligibility.
5. If TTP is enabled, checks service health and sends the exact source bytes in earlier/later order.
6. Validates the returned mask, dimensions, binary values, provenance, and statistics.
7. Recomputes the shared connected-component statistics from the learned mask.
8. Makes TTP statistics and mask primary; deterministic evidence remains independently labelled support.
9. Produces learned, deterministic, agreement, disagreement, intersection, union, and overlay previews.
10. Uses the primary statistics for controlled change questions and report generation.

## 3. Current inference path

`TTPServiceClient.predict()` sends multipart fields `earlier_image` and `later_image` to `/predict`. The service validates and decodes both inputs, then calls `ModelLifecycle.predict()`, which serializes access to one `OpenCDInferencer`. The inferencer receives `[[earlier_path, later_path]]` and returns a prediction that is unwrapped and converted to a strict two-dimensional Boolean mask.

The main backend never loads the checkpoint directly. It consumes the service's opaque binary mask artifact and rejects malformed provenance, dimensions, values, statistics, content types, or artifact references.

## 4. Checkpoint loading path

The service defaults are:

- repository: `/opt/ttp/TTP`, override `TTP_REPOSITORY_DIR`;
- checkpoint: `/opt/ttp/assets/epoch_260.pth`, override `TTP_CHECKPOINT_PATH`;
- config: `/opt/ttp/TTP/configs/TTP/ttp_sam_large_levircd_infer.py`.

`download_assets.py` acquires the official repository and checkpoint. `verify_assets.py` requires the pinned repository commit, a 1.2–1.35 GB checkpoint, and the exact SHA-256 before initialization.

## 5. Checkpoint lifecycle

`ttp_service.lifecycle.MODEL_LIFECYCLE` is process-global. It retains one inferencer and uses separate load and inference locks. Existing counters are model load count, inference count, reuse count, load latency, and CUDA allocated/reserved/peak memory.

Current gaps:

- startup loading defaults to enabled instead of true first-use lazy initialization;
- `/predict` refuses an unloaded lifecycle instead of loading it lazily;
- a failed lifecycle can attempt loading again, conflicting with a sticky fail-safe lifecycle;
- the main backend health response does not expose the service's checkpoint, loaded flag, load count, reuse count, or safe error state.

## 6. Current preprocessing

SatQuery sends exact validated source bytes and performs no TTP resize, registration, reprojection, or resampling. The isolated service validates image signatures, sizes, modes, and equal dimensions, converts inputs to RGB for validation/overlay, and passes temporary source files to the official `OpenCDInferencer`.

Model resize, tensor conversion, and normalization are delegated to the pinned official inference config. They are not duplicated or reimplemented in SatQuery. The official repository/config is absent on this host, so its exact numeric normalization cannot be independently inspected here; production correctness therefore depends on verifying the pinned repository before model load.

## 7. Expected image order

The earlier/before image is always first and the later/after image second:

- API: `before_image`, `after_image`;
- client multipart: `earlier_image`, `later_image`;
- inferencer input: `[[earlier_path, later_path]]`.

No code reverses or auto-sorts the pair.

## 8. Expected normalization

No application-defined ImageNet or joint-percentile normalization is applied to TTP. The official TTP/OpenCD test pipeline from `ttp_sam_large_levircd_infer.py` is authoritative. The deterministic analyzer separately uses joint 2nd/98th-percentile normalization; that normalization is not fed into TTP.

## 9. Output tensor

The current adapter unwraps NumPy arrays, tensors, `pred_sem_seg`, dictionaries, lists, and data samples. It accepts either a two-class tensor (argmax) or a two-dimensional binary prediction. Values must resolve to finite `{0,1}` output, with `[0,1]` floating values thresholded at `0.5`.

The current production boundary exposes only the binary mask. It does not explicitly extract an internal probability map before thresholding, so probability-map execution cannot presently be demonstrated in trace data.

## 10. Current API integration

Both existing change entry points call `_change_query_result()`. Public request and response models already contain optional `change_engine`, `ttp_result`, `deterministic_statistics`, `mask_comparison`, `evidence_consistency`, and source-labelled preview fields. A successful learned response sets `statistics` from the TTP mask and marks `ttp_change_detector` as primary.

No new endpoint or router path is required.

## 11. Current health endpoint

The isolated `/health` exposes lifecycle, checkpoint verification fingerprint, device, model load count, inference count, reuse count, and limitations. The main `/api/agent/health` reduces this to `status`, `device`, and `error`, and reports only `disabled` or `unavailable` when TTP is not ready. It omits checkpoint name, loaded state, counters, and safe error details required for production diagnosis.

The root backend `/health` does not include SatQuery specialist health; the specialist contract is `/api/agent/health`.

## 12. Current execution trace

On success the main execution summary may contain:

- upload and metadata stages;
- pair validation;
- deterministic analysis and its five stages;
- TTP eligibility and health checks;
- request preparation;
- model load/reuse;
- TTP inference;
- mask validation;
- mask comparison and evidence fusion;
- controlled answer and response generation.

The service also returns its own validation/model/inference/mask trace, but `TTPClientResult.service_trace` is not copied into the public execution summary. Checkpoint, runtime, and device appear in `ttp_result`; fallback and mask-generation state are spread across other existing fields rather than consistently attached to TTP execution-step parameters.

## 13. Current fallback behavior

The deterministic analyzer runs for eligible aligned pairs before TTP. Disabled TTP, unavailable or unhealthy service, timeout, CUDA OOM, checkpoint/provenance mismatch, malformed response, invalid mask, invalid artifact, or statistics mismatch all preserve a successful deterministic result. `change_engine` records `deterministic_fallback` and a bounded reason. Warnings disclose the fallback without leaking paths or remote exceptions.

## 14. Missing production connections

- TTP is default-off in code and `.env.example`.
- No TTP service is listening on the configured local port.
- No official TTP repository or `epoch_260.pth` exists on this host.
- Service inference is startup-oriented, not lazy on first valid request.
- Main agent health does not expose the complete learned-engine lifecycle.
- Main execution steps do not carry all required checkpoint/device/fallback/mask state.
- Service trace is validated but not propagated into the existing execution-step format.
- The internal probability-map-to-binary-mask stage is not explicit.
- There is no production smoke runner that asserts real learned execution, reuse, mask/statistics/report contracts, memory, and health together.
- The exact phrases `Where did change occur?` and `How much changed?` are not currently claimed by the unchanged router rules (the existing controlled variants are more specific). The task explicitly forbids routing changes, so this integration cannot broaden those phrases without separate authorization.

## 15. Blocking issues

1. **Runtime assets:** `/opt/ttp/TTP` and `/opt/ttp/assets/epoch_260.pth` are absent, and no cached `epoch_260.pth` was found under the project, Desktop, home directory, or Hugging Face cache.
2. **Compute/runtime:** the official service is pinned to Python 3.10, CUDA 12.1, PyTorch 2.1.2, and OpenCD/MMCV dependencies. This macOS host has no CUDA runtime and cannot execute the verified production checkpoint.
3. **Service availability:** nothing is listening on `127.0.0.1:8000`.
4. **Probability map:** the adapter consumes final segmentation output and does not prove an explicit probability-map stage.
5. **Default activation:** `TTP_ENABLED=false` prevents the client from attempting learned inference.
6. **Observability:** the main health and execution trace omit required lifecycle/provenance detail.
7. **Frozen routing vocabulary:** two requested example phrasings are outside the existing rule list, and routing changes are expressly forbidden by this task.

Production code can close the lifecycle, activation, observability, trace, probability handling, test, and smoke-runner gaps. A successful real-checkpoint smoke run still requires deployment of the verified pinned assets on a compatible CUDA host; the deterministic fallback must remain authoritative until that external prerequisite is met.
