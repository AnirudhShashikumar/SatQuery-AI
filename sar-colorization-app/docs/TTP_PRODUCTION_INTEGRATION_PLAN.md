# TTP production integration plan

## Existing

Validated bi-temporal requests run deterministic change analysis first. When explicitly enabled and a separately deployed verified CUDA service is ready, `TTP_CLIENT` requests a learned binary mask, validates it, promotes it to primary evidence, and retains deterministic evidence as support. The default configuration disables TTP, service inference is not first-use lazy, main health is lossy, and no production smoke runner verifies the complete learned path.

## Desired

Validated aligned optical pair

→ unchanged router and input validator

→ unchanged alignment eligibility

→ singleton TTP client and singleton CUDA model lifecycle

→ lazy verified checkpoint load exactly once

→ official TTP preprocessing in earlier/later order

→ internal probability map and fixed `0.5` binary decision

→ validated learned mask as primary evidence

→ existing connected components, statistics, previews, VQA, reports, and analytics

→ independently labelled deterministic supporting evidence

→ automatic deterministic fallback for every learned-path failure

## Files to modify

- `satquery_agent/specialists/ttp_change.py`: retain the singleton client, add safe complete health/lifecycle introspection, preserve strict service response and artifact validation, and retain bounded fallback errors.
- `satquery_agent/api.py`: use the complete health payload and enrich existing execution steps with checkpoint, device, runtime, load/reuse, fallback, and mask state without changing models or response structure.
- `ttp_service/lifecycle.py`: make first valid prediction lazily load the verified model once, keep failures sticky, expose lifecycle introspection, and preserve serialized inference.
- `ttp_service/inference.py`: make probability-map extraction and the `0.5` binary decision explicit while retaining official OpenCD preprocessing and model architecture.
- `ttp_service/app.py`: default startup preloading off, allow lifecycle prediction to perform the lazy load, and record probability-map generation in the existing trace list.
- `satquery_agent/registry.py`: clarify verified checkpoint lifecycle, probability-to-mask behavior, primary learned evidence, and CUDA limitations.
- `.env.example` and `README.md`: make the learned hybrid engine the production default while documenting the exact rollback flag and external CUDA prerequisites.
- `scripts/run_ttp_production_smoke.py`: verify real API execution, mask/statistics, trace, health, singleton counts, reuse, runtime, memory, and report generation.
- focused backend and service tests: lifecycle, health, probability thresholding, fallback, checkpoint failure, API execution, trace, statistics, report compatibility, and response-schema preservation.

`router.py`, public request/response models, endpoints, frontend contracts, model weights, checkpoint contents, and analytics schemas will not change.

## Tests

1. Lazy first prediction loads once.
2. Repeated prediction reuses the same inferencer.
3. Failed checkpoint load is sticky and does not reload.
4. Logits become a finite changed-class probability map and a fixed binary mask.
5. Existing final segmentation outputs remain backward compatible.
6. Main health includes required TTP lifecycle fields without changing `AgentHealth`.
7. Successful TTP API execution makes the learned mask primary and preserves deterministic support.
8. Unavailable service, timeout, malformed mask, and checkpoint failure preserve deterministic results.
9. Execution trace contains checkpoint, device, runtime, load/reuse, fallback, and mask state.
10. Existing report generation consumes the unchanged response contract.
11. Router and response schema snapshots remain unchanged.

## Smoke tests

The production smoke runner will submit the checked-in aligned optical pair twice with `What changed?` to the existing `/api/agent/query` endpoint, fetch and validate the binary mask, validate learned statistics and execution steps, query `/api/agent/health`, verify one model load plus reuse, record process/GPU memory reported by the deployment, and request an existing report artifact. It will fail clearly if the learned checkpoint did not execute; a deterministic fallback is valid production behavior but is not a passing learned-engine smoke result.

## Rollback strategy

Set `TTP_ENABLED=false` or `TTP_DEFAULT_MODE=deterministic` and restart the backend. This restores deterministic-only execution without a code rollback, model mutation, route change, or response-contract change. The isolated CUDA service can then be stopped independently. Existing deterministic fields, previews, controlled answers, reports, and analytics remain operational.

## Backward compatibility guarantees

- Existing endpoints, multipart field names, router rules, task types, Pydantic response models, execution-step structure, frontend contracts, and report formats remain unchanged.
- Existing deterministic analysis remains present and is the automatic fallback.
- No image is automatically aligned, registered, reprojected, or resized by SatQuery.
- No checkpoint or model weight is modified.
- Existing TTP service clients remain compatible because service response models and artifact endpoints are unchanged.
- New health details are additive entries inside the existing `specialists: Dict[str, Any]` container.
