# Optical + SAR root-cause audit

## Observed failure and expected behavior
The supplied run describes SAR-preview content in the Optical slot and RGB content in the SAR slot, an accepted pair, a mislabeled SAR source, skipped preparations, and a registration-only refusal. Correct behavior is to reject slot/content conflicts explicitly, retain source identity, and allow disclosed qualitative preview analysis while blocking unverified spatial metrics.

## End-to-end mapping and relevant files
- `frontend/components/assistant-workspace.tsx`: primary file is displayed in the Optical slot; secondary in SAR. `updateSecondary` changes workflow from filename hints. The inspection effect replaces primary/secondary modality state with detected modalities and accepts an unordered optical/SAR set, without changing fixed slot labels or checking their roles.
- `frontend/services/api.ts`: multipart named primary/secondary fields retain their file identities (no evidence of multipart reordering).
- `satquery_agent/api.py`: `/query` ingests the named fields in order. Legacy coarse primary declarations can override effective modality. Pair validation checks declarations, not detected content. `_order_cross_modal_inputs` reorders according to declarations; `/cross-modal` also trusts declarations despite named slots.
- `satquery_agent/image_ingestion.py`: content detection is separate and already detects normal chromatic RGB and hinted, textured near-grayscale SAR previews. Grayscale optical filename hints can produce unwarranted certainty. No evidence that the actual inputs were misdetected; their displayed classifications suggest correct detection.
- `satquery_agent/compatibility.py`: checks an unordered declared modality set and independently classifies geospatial compatibility.
- `satquery_agent/specialists/cross_modal.py`: non-EXACT pairs return before preparation/extraction. Native optical and SAR extraction already exist but are unreachable for PNG pairs. No translation dependency exists here.
- `satquery_agent/specialists/vqa.py`: missing statistics always yields the pixel-level refusal, regardless of qualitative intent.
- `satquery_agent/api.py`: preparation trace success and evidence publication are conditional on statistics, conflating source evidence with joint spatial metrics.
- `frontend/components/assistant-result-experience.tsx`: named cross previews take precedence, but fallbacks label primary as Optical and secondary as SAR. These assumptions can invert legacy declared-order results. Products lack observation identity.
- `satquery_agent/reporting.py`, `comparison.py`, and `models.py`: retain named previews and primary/secondary metadata but lack explicit observation-role/product identity for cross-modal provenance.
- `backend.py` includes the agent router; SAR preprocessing and native scene/water modules preserve numeric-domain safeguards. They need no model or checkpoint changes.

Browser verification additionally exposed a viewer-state contributor: it selected the second product with an opacity blend (SAR over optical) by default, yet displayed only the SAR label. Source tabs now force a pure layer; user-selected blends/splits name both sources. The original reported visual could therefore be caused by blending as well as a stale role declaration.

## Exact causes and limits of attribution
The reproducible defects are role/declaration conflation, unordered validation, index-based viewer fallback, and an overbroad registration gate. The existing browser effect can show a reversed pair as valid, while a stale/explicit declaration can feed the optical image into the SAR analyzer and thus publish it as `previews.sar`. No original request payload is available, so the precise runtime sequence that produced the supplied PDF cannot be proven. Multipart ordering itself is not the cause found in code.

## Why tests missed it
Cross-modal tests submit declared-order exact GeoTIFFs. The PNG test explicitly expects partial/no fusion. Frontend tests use consistent named preview fixtures; they do not connect reversed upload inspections to submission or test contradictory index fallbacks. Geospatial tests correctly protect alignment, but do not distinguish qualitative intent.

## Scientific implications
An accepted role mismatch corrupts every downstream source claim. Conversely, lack of registration blocks spatial intersections, coordinates and joint percentages, but does not prevent independent native-source evidence or coarse image-relative comparison. Radar darkness/brightness alone cannot establish water/building identity.

## Proposed repair
Introduce explicit roles and validate them against detected content at ingestion, before cache lookup or specialists. Preserve legacy declared order only when explicit declarations and detected content agree; the current UI sends semantic slot roles. Normalize into a named pair. Publish source IDs, roles, modalities and evidence types. Keep geospatial classification unchanged. Add a separate same-size, unreferenced PNG/JPEG qualitative path, comparing independently extracted evidence in coarse image-relative sectors without intersecting masks. Keep exact-grid spatial fusion protected. Synthesize query-aware qualitative findings and distinguish unavailable requested metrics from completed qualitative answers. Record preparation, native extraction, fusion and quantitative eligibility independently. Preserve old report loading with additive optional fields.

## Backward compatibility
Named and legacy multipart fields remain supported. Explicit legacy reversed declarations remain supported when content matches them; implicit reversed content is rejected. Stored responses remain readable; old ambiguous source fallbacks must not invent roles. Cross-modal cache version changes prevent stale refusals or mislabeled products from being reused. Existing exact-grid algorithms, model artifacts and benchmark scores remain unchanged.

## Test plan
Extend existing ingestion, cross-modal, scientific compatibility, compliance, report/comparison, upload and result-viewer suites. Include 256×256 corrected/reversed PNG fixtures, ambiguous inputs, misleading filenames, both API entry points, explicit roles and legacy order, qualitative query variants, quantitative refusals, exact/shifted/CRS/non-overlap grids, missing/optional specialist behavior, source previews, trace, JSON/PDF/ZIP and legacy reads. Run focused backend tests, full pytest, frontend tests, TypeScript, ESLint, production build, compliance, final submission audit and available real smoke tooling. Record actual results and limitations below after implementation.

## Final implementation and verification report

1. **Root cause:** slot roles, detected modality and legacy declarations were conflated; validation accepted an unordered modality set. Registration eligibility also incorrectly controlled all native-source preparation and answering.
2. **Why modalities appeared reversed:** the UI changed modality state without changing slot labels, the backend trusted declarations, and the viewer used positional fallbacks. Browser verification also reproduced the misleading default SAR-over-optical blend labeled only “SAR source.” The original request payload remains unavailable, so its exact runtime sequence is not claimed as proven.
3. **Exact files changed:** see the complete source/test/documentation list below. Generated verification records and smoke exports are listed separately.
4. **Backend mapping:** `ObservationRole` metadata is attached at ingestion. A named `CrossModalPair` resolves observations by validated role. Both APIs distinguish detected content from roles. New UI multipart requests send explicit slot roles. Legacy explicitly reversed declarations are supported only when content agrees; slot/content mismatches and unknown content are blocked before specialists/cache reuse.
5. **Frontend mapping:** selected workflows no longer change from filename hints or an unordered detection effect. The Optical and SAR slots retain fixed roles; reversed pairs are blocked with an explicit swap action that switches the actual files and re-inspects both. Demo loading inspects both sources. Switching workflows retains inspection for unchanged files.
6. **Detection:** normal chromatic RGB remains optical even under a SAR filename. Near-grayscale SAR previews still require corroborating display statistics and hints. An optical filename no longer overrides conflicting SAR-like display statistics; such inputs remain unknown. Scientific metadata/band interpretation stays authoritative under the existing ingestion rules.
7. **PNG/JPEG:** valid same-size unreferenced preview pairs can finish supported qualitative questions with `success` / `COMPLETED`, independent optical/SAR findings, coarse image-relative agreements, complementary findings and the explicit registration disclosure. Neither pixel intersections nor joint percentages are computed.
8. **GeoTIFF protection:** the exact-grid quantitative algorithm is unchanged. Different CRS, shifted grids, different-size inputs and insufficient/no overlap retain existing scientific classifications and protected outcomes. No reprojection, registration or silent resampling was added.
9. **Optical preparation:** native normalization, RGB mapping provenance and independent visible/edge/texture candidates execute for eligible previews. Optional SVE evidence retains its existing provenance. Low-information optical inputs cannot produce qualitative agreement from SAR alone.
10. **SAR preparation:** the native analysis raster is used directly, including the existing single-channel SAR-preview extraction. Relative intensity, smoothness and structural/texture candidates do not require translation or fabricated polarization. Calibration is not inferred.
11. **Native SAR evidence:** it is a primary source with observation ID, role, detected modality, evidence type and stable per-response evidence ID. The result viewer and source cards show native optical and SAR evidence separately.
12. **Fusion:** qualitative sector summaries compare independent candidate support. Query-aware synthesis handles water, structure, agreements, disagreements, complementary SAR evidence, scene descriptions and the official combined query. Confidence is low/moderate with no calibrated probability. A missing native extractor produces a failed result, no fusion and a visible error trace.
13. **Evidence viewer:** source tabs resolve provenance before legacy named-preview fallback; they never infer optical/SAR from primary/secondary array position. Source tabs force a single-image layer. User-selected blends/splits name both images. Tests check the actual rendered image element and URL.
14. **Reports/exports:** source roles and IDs are serialized alongside primary/secondary compatibility fields. JSON/ZIP include role-indexed input metadata and evidence products; PDF input headings and source-image labels retain identities. Comparison previews carry source identity. Additive optional fields preserve old result parsing; absent legacy role information is not invented.
15. **Trace:** optical preparation, SAR preparation, native extraction and qualitative fusion report success separately from quantitative eligibility. Spatial eligibility and region extraction explicitly skip without registration; the reason is visible in the frontend. Quantitative requests on qualitative pairs finish partial with limitations.
16. **Tests added/extended:** 256×256 corrected/reversed PNG fixtures; both API entry points; seven qualitative intent variants; quantitative refusals; source IDs/roles and legacy order; native extractor failures; optional-translation independence; misleading filenames/ambiguity; low-information optical protection; JSON/PDF/ZIP mapping; source-only viewer rendering; explicit swap/reinspection; demo inspection and workflow switching. Existing geospatial, five-query PS, registry, model and provenance tests still execute. Old fixtures that passed RGB as SAR or assumed filenames could override content were corrected rather than bypassed.
17. **Focused backend:** 142 passed, 18 warnings. Suites: cross-modal, scientific pair compatibility, ingestion, reporting/readiness, comparison, single-image SAR and PS compliance.
18. **Full backend:** 598 passed, 5 skipped, 27 warnings. Skips remain existing environment/checkpoint-dependent tests; none were added to bypass this repair.
19. **Frontend:** 199 passed across 33 files.
20. **TypeScript:** `npx tsc --noEmit --incremental false` passed.
21. **ESLint:** passed with 0 errors and 27 warnings, primarily existing `no-img-element` advisories; this is not a warning-free result.
22. **Production build:** passed. The final local backend and production frontend were reloaded and used for the final browser check.
23. **PS 26167:** standalone compliance suite: 10 passed. Final audit's combined benchmark-schema/compliance gate: 16 passed. All five representative queries retain their expected workflow contracts.
24. **Final submission audit:** `final_submission_audit.py --run-tests` passed, including full backend/frontend tests, typecheck, lint, build, benchmark validation, checkpoint verification and existing all-workflow smoke. A pre-existing personal path in launcher documentation was replaced with a portable HOME-based example to satisfy its private-path scan. `git diff --check` passed.
25. **Real smoke and remaining limitations:** the existing `optical 1.png` and `sar 1.png` images were run in the actual browser. Reversed slots were blocked; explicit swapping restored correct identities and enabled analysis. Final run `68b1e9ec-0403-45df-8b95-47906f9c9246` completed in 1,611 ms with native evidence, successful preparations, a qualitative answer and the registration disclosure. Fullscreen visual inspection confirmed optical RGB under Optical source and grayscale radar under SAR source, each as one image. UI health showed 6/6 specialists ready; two unrelated optional services were unavailable. The reusable real-image in-process smoke also passed, including quantitative refusal and JSON/PDF/ZIP exports. All seven rendered PDF pages were inspected for source mapping, legibility and clipping. Qualitative sector co-occurrence remains a heuristic with false positives, not verified pixel agreement or semantic truth. Ambiguous previews require replacement. Optional translation is not invoked by this native-pair workflow; existing translation workflows remain supporting only. Corrupt historical provenance cannot be reconstructed without the original request. The installed test environment is Python 3.9 despite the documented Python 3.11 target; existing dependency warnings remain recorded.
26. **Integrity statements:** model checkpoints changed? **no**. Model logic changed? **no** for neural architecture, weights and model inference; deterministic validation, evidence-fusion and answer logic **did change**, as required by the repair. Benchmark numbers changed? **no**. SIH safeguards weakened? **no**. Hardcoded answers added? **no**; prose is synthesized from computed findings. Geospatial safeguards bypassed? **no**. Only the models-contract source hash and frontend source digest in the existing ChangerEx integrity manifest were refreshed for authorized source edits; checkpoint, TTP and other protected model hashes were not changed, and the integrity tests remain active.

## Complete changed-file list

Paths are relative to this application directory. The user’s pre-existing sample PDF change is outside this repair.

- `satquery_agent/models.py`
- `satquery_agent/api.py`
- `satquery_agent/compatibility.py`
- `satquery_agent/image_ingestion.py`
- `satquery_agent/specialists/cross_modal.py`
- `satquery_agent/specialists/vqa.py`
- `satquery_agent/reporting.py`
- `satquery_agent/comparison.py`
- `frontend/components/assistant-workspace.tsx`
- `frontend/components/assistant-result-experience.tsx`
- `frontend/components/satquery/evidence-sources.tsx`
- `frontend/services/api.ts`
- `frontend/types/agent.ts`
- `frontend/components/assistant-workspace-ui.test.tsx`
- `frontend/components/assistant-result-experience.test.tsx`
- `tests/test_cross_modal.py`
- `tests/test_satquery_ingestion.py`
- `tests/test_single_image_sar.py`
- `tests/changerex_protected_hashes.json` (source-code baselines only)
- `scripts/run_optical_sar_preview_smoke.py`
- `docs/OPTICAL_SAR_ROOT_CAUSE_AUDIT.md`
- `docs/OPTICAL_SAR_EVALUATION_PROTOCOL.md`
- `docs/PROBLEM_STATEMENT_26167_COMPLIANCE.md`
- `docs/MACOS_APP_LAUNCHER.md`
- `README.md`

Generated verification records:

- `artifacts/final_readiness/optical_sar_submission_audit.json`
- `artifacts/final_readiness/optical_sar_workflow_smoke.json`
- `artifacts/final_readiness/workflow_smoke.json`
- `artifacts/final_readiness/optical_sar_preview_smoke/`: `smoke.json`, `response.json`, `reversed-validation.json`, four role-labeled native/source PNGs, `mission-report.json`, `mission-report.pdf`, `mission-report.zip`.

Reproduce the real-image smoke with `venv/bin/python scripts/run_optical_sar_preview_smoke.py --optical PATH_TO_OPTICAL --sar PATH_TO_SAR --output artifacts/optical_sar_preview_smoke`. It uses actual uploads and the production API in-process, with offline model loading; it never turns a smoke result into benchmark accuracy.
