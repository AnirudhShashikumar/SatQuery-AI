# SatQuery AI — SIH Final Engineering Audit

Audit date: 2026-09-03  
Problem statement: ISRO / Department of Space, SIH 26167  
Scope: repository state before this final engineering-readiness pass

## Audit method

The audit inspected the FastAPI entry points and typed contracts, router, specialist registry, image ingestion and pair compatibility code, all production specialists, reporting and benchmark schemas, canonical benchmark records, frontend routes/components/types, existing documentation, and backend/frontend tests. This document is intentionally written before implementation changes. It distinguishes operational behavior, verified benchmark evidence, and missing evaluation evidence.

## Requirement-to-implementation gap map

| Official requirement | Existing implementation | Gap | Proposed change | Files affected | Risk | Test strategy |
|---|---|---|---|---|---|---|
| Single optical/multispectral/SAR analysis | Typed ingestion; optical VQA/caption/grounding/SVE; native SAR water/scene/translation paths | Unknown 4+ band rasters can still reach display preparation without a complete explicit visual-band decision record | Add conservative band-selection utility and optional provenance fields; reject semantic RGB inference when mapping is unknown | `image_ingestion.py`, `models.py`, focused ingestion tests | Medium: compatibility with legacy TIFFs | Synthetic RGB, metadata-labelled Sentinel-like, unknown 4-band, and legacy-result tests |
| Optical + SAR joint reasoning | Exact-grid deterministic optical/native-SAR fusion with regions, previews and confidence rationale; optional translated-SAR support exists for single-SAR workflow | Pair answer/provenance is summary-list based rather than fact-level; query-aware response coverage and failure combinations are incomplete | Add optional source-labelled facts and conservative query-aware synthesis without changing endpoint schema requirements | `models.py`, `specialists/cross_modal.py`, API orchestration tests | Medium: response wording and legacy clients | Water, built-up, agreement, disagreement, unavailable branches, translation disclosure, provenance tests |
| Bi-temporal change and VQA | ChangerEx default learned engine, independent deterministic evidence, mask/regions/overlays, safe fallback; local semantic interpreter and optional `semantic_change_summary` already implemented | Verify full prompt matrix, trace metadata, malformed evidence, and public regression coverage; do not rebuild | Extend focused tests and compliance regression | semantic specialist/API tests, `tests/compliance/` | Low | Claim-gate, location, magnitude, disagreement, fallback, old-result tests |
| PNG/JPEG/TIFF/GeoTIFF | Byte-sniffed ingestion, bounded reads, Rasterio metadata, previews | Compatibility enum is coarse (`exact`, `overlap`, `visual_only`, `incompatible`); resolution/orientation/NoData details are not explicit | Add optional scientific compatibility classification/details while preserving old fields | `models.py`, `compatibility.py`, ingestion fixtures | Medium: geospatial edge cases | Exact/shifted/different CRS/resolution/partial/no overlap/missing CRS/PNG/malformed metadata |
| Remote-sensing adaptation | SVE BigEarthNet-derived adapter; RSVQA specialist; RSICD captioner; VRSBench grounding head | Provenance is distributed and license completeness varies | Create auditable inventory; mark unknown licenses explicitly | provenance documentation, model metadata readers | Low | Documentation/path scanner and checkpoint verification tooling |
| Agentic orchestration | Deterministic task classification, validation, registry, specialist execution, confidence, evidence and trace | Registry lacks typed allowed-parameter constraints and centralized validation | Extend registry contracts with optional parameter schemas/defaults/constraints and safe validator | `models.py`, `registry.py`, `router.py`, tests | Medium: routing compatibility | Defaults/type/range/unknown-field/trace tests |
| Evidence-grounded outputs and reports | Preview/mask/overlay/regions; PDF/JSON/CSV/ZIP; confidence disclaimers | Final problem-statement regression does not cover all five representative queries end-to-end | Add data-independent compliance regression fixtures | `tests/compliance/`, compliance docs | Low | Assert workflow/task/specialists/evidence/trace/provenance, never exact prose |
| RSVQA evaluation | Verified official test record: 10,004 questions, 71.05% exact, 74.31% overflow-aware | None for the verified record; retain unchanged | Validate canonical record only | benchmark validation tests/docs | Low | Schema validation and immutable-values regression |
| VRSBench grounding evaluation | Verified validation record: 16,146 samples, mIoU 0.2348, Acc@0.50 25.05%; Smoke-100 regression artifacts | None for the verified record; retain unchanged | Validate canonical record only | benchmark validation tests/docs | Low | Schema validation and immutable-values regression |
| CDVQA evaluation | No complete evaluator or local dataset evidence | Full evaluator and official protocol adapter absent | Build dataset-validating evaluator using production bi-temporal path; emit unavailable/error clearly when data absent | `scripts/evaluate_cdvqa.py`, docs/tests | Medium: public dataset variants | Synthetic manifest adapter, split preservation, missing-data and normalization tests |
| ChangerEx LEVIR-CD evaluation | Pure-PyTorch official extraction, parity, operational MPS/CPU smoke; quality benchmark pending | Official-test evaluator absent | Build SHA-verified full evaluator with global/per-image metrics and status gates | `scripts/evaluate_changerex_levircd.py`, tests/docs | Medium: dataset/checkpoint may be absent | Metric fixtures, hash mismatch, incomplete split, no silent skips |
| Controlled optical+SAR evaluation | Canonical unavailable benchmark record | Protocol/evaluator absent | Add task-specific three-path evaluator and fusion-gain guard | evaluator module/script, protocol doc, tests | Low | Same-sample/same-metric guard and unavailable-data behavior |
| SAR translation evaluation | Pix2Pix operational smoke and SARFusionFormer unavailable record | Common paired evaluator absent; LAB-to-RGB regression not centralized | Add paired evaluator, correct LAB conversion gate, and unavailable status | script/tests/docs | Medium: model adapters and LPIPS optionality | Known LAB/RGB fixture, metric calculation, missing dependency/data behavior |
| Caption evaluation | Functional captioner; canonical unavailable record | Evaluator/reference adapter absent | Add reference-caption evaluator with only installed/computable metrics | script/tests/docs | Low | Tokenization/reference fixtures and unavailable references |
| Confidence calibration | Explicit uncalibrated disclosures | No offline calibration utility | Add optional ECE/Brier/reliability tool; never relabel runtime scores | script/tests | Low | Known-bin fixtures and insufficient-evidence case |
| SIH compliance view | Backend compliance endpoint and data-driven frontend page | Status vocabulary is legacy `available`; benchmark evidence is not first-class | Add optional status/evidence fields and conservative mapping without breaking clients | `models.py`, `compliance.py`, frontend compliance types/component/tests | Medium | Old payload compatibility and no-zero-for-missing tests |
| Input workspace | Strong landing page plus compact workflow identity, paired cards, prompts, two-column layout, neutral demos, collapsed technical nav already implemented | No material gap found | Verify only; preserve current UI | existing frontend tests/CSS | Low | Unit suite, TypeScript, build, 1366/1440/1536/1920 and mobile visual review |
| Model/dataset provenance and licenses | Several model cards and integration audits | No consolidated complete inventory; some licenses not verified | Create `MODEL_DATASET_PROVENANCE.md`; unknown remains “license verification required” | docs | Low | Link/path/checkpoint scanner |
| Clean install/offline demo | README and per-component docs exist | No single offline verifier/checklist covering all workflows | Add non-mutating installation/model/workflow verification utilities and checklist | scripts/docs/tests | Low | Temp repository fixtures and offline mode |
| Performance measurement | Specialist-specific smoke/latency artifacts exist | No unified workflow runtime benchmark | Add repeatable performance harness, clearly separate from accuracy | script/docs/tests | Low | Synthetic/no-model adapter and schema tests |
| Hidden Cartosat-2S/RISAT readiness | Generic TIFF dtype/statistics and SAR modality logic | Sensor-domain assumptions and unverified status not consolidated; SARFusionFormer requires explicit VV/VH enforcement | Harden metadata/provenance, document risks, forbid silent channel duplication | ingestion/SAR preprocessing/tests/docs | Medium | uint8/uint16/float32, one/two-band, missing polarization, unknown range |
| Final submission readiness | Broad backend/frontend suites and benchmark schema validation exist | No single final audit checklist/script | Add read-only final audit script and document | scripts/docs/tests | Low | Fixture repo checks and full suite |

## Current architecture and contracts

- Public backend contracts are Pydantic models in `satquery_agent/models.py`; additive optional fields are the only safe schema evolution in this pass.
- `/api/agent/*` endpoints preserve request/response shapes and feed the process-local report store. Existing report extraction already tolerates optional semantic change content.
- The router is deterministic and exposes an observable routing reason. It must not expose hidden reasoning or accept arbitrary checkpoints.
- ChangerEx is the default local learned bi-temporal engine; the deterministic analyzer remains independent support and fallback. TTP remains optional.
- Cross-modal analysis currently requires exact pixel alignment for fused maps. Non-georeferenced benchmark pairs are reported as visual-only and are not called georeferenced.
- Canonical benchmark JSON is source-of-truth for the frontend. Missing quality evidence is represented as unavailable, never zero.

## Existing verified evidence that must remain unchanged

| Specialist | Dataset/split | Samples | Evidence status | Metrics |
|---|---|---:|---|---|
| RSVQA Specialist v1 | RSVQA-LR official test | 10,004 | verified test | 71.05% exact; 74.31% overflow-aware |
| Grounding Specialist v1.1 | VRSBench validation | 16,146 | verified validation | mean IoU 0.2348; Acc@0.50 25.05% |
| ChangerEx | Local parity and production smoke only | — | operational only | No quality metric claimed |
| Pix2Pix | Single-image smoke only | 1 | operational only | No paired-target quality metric claimed |
| SARFusionFormer | No common held-out evaluation | — | unavailable | No quality metric claimed |
| Captioner | No reference-caption evaluation | — | unavailable | No quality metric claimed |
| Optical+SAR fusion | No controlled labelled evaluation | — | unavailable | No fusion accuracy claimed |

## Implementation order and stop conditions

Implementation will follow the requested phase order. A phase already satisfied will receive verification and documentation instead of redundant code. Missing public datasets, official split ambiguity, absent checkpoints, or unverified licenses will produce safe evaluator infrastructure plus an explicit evidence gap; no result will be fabricated. No checkpoint will be modified, no model will be retrained, and no external service will become mandatory.
