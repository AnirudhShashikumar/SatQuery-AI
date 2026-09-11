# Problem Statement 26167 Compliance

Status date: 2026-09-03. Status labels distinguish implementation from public-dataset verification. Missing benchmark evidence is never represented as zero.

| Official requirement | SatQuery implementation | Specialist(s) | Automated coverage | Benchmark evidence | Status |
|---|---|---|---|---|---|
| Single optical/multispectral image VQA | Validated ingestion, controlled VQA, evidence and trace | RSVQA Specialist v1; deterministic VQA fallback; SVE support | representative-query suite; VQA/API/fallback tests | RSVQA-LR official test, 10,004 questions: 71.05% exact, 74.31% overflow-aware | VERIFIED |
| Single SAR analysis | Native intensity, texture, water-candidate and quality analysis | SAR Scene Analyzer; SAR Water Specialist | SAR ingestion, preprocessing, water and scene tests | No RISAT-labelled quality benchmark | IMPLEMENTED |
| Captioning | Local RSICD-adapted optical captioner | Remote-sensing captioner; optional SVE consistency | captioner and API tests | Reference-caption benchmark pending | PARTIALLY VERIFIED |
| Text-guided grounding | Grounding DINO proposals with learned VRSBench rescorer and quality gates | Grounding DINO Tiny; Grounding Specialist v1.1 | grounding loading/fallback/box/API tests | VRSBench validation, 16,146 samples: mIoU 0.2348, Acc@0.50 25.05% | VERIFIED |
| Bi-temporal change mask/map | ChangerEx learned mask with independent deterministic evidence and safe fallback | ChangerEx; Deterministic Change Analyzer; optional TTP | lifecycle, integration, parity, mask and fallback tests | Official LEVIR-CD quality run pending; parity/operational smoke is not accuracy | PARTIALLY VERIFIED |
| Change description and VQA | Offline structured semantic summary with explicit evidence claim gate | Bi-temporal Semantic Interpreter | semantic direction/location/disagreement/fallback/legacy tests | CDVQA evaluation pending | IMPLEMENTED |
| Optical + SAR joint extraction | Exact-grid optical and native-SAR evidence branches with fused regions and previews | Optical-SAR Joint Analyzer; optional SVE/translation support | cross-modal endpoint and pair validation tests | Controlled labelled evaluation pending | IMPLEMENTED |
| GeoTIFF/TIFF and PNG/JPEG | Byte-validated bounded ingestion, Rasterio metadata, scientific/display separation | Input Validator | ingestion and synthetic raster tests | Operational validation only | IMPLEMENTED |
| Pair compatibility | CRS/transform/bounds/dimensions/overlap checks; no silent reprojection/resampling | Input Validator | synthetic pair tests | Not an accuracy claim | IMPLEMENTED |
| Remote-sensing adaptation | BigEarthNet-derived SVE, RSVQA/VRSBench task heads, RSICD captioner | SVE and task specialists | checkpoint/provenance tests | Task-specific records above | VERIFIED |
| Agent orchestration | Deterministic intent classification, compatibility validation, registry selection and bounded parameters | Router and registry | representative-query and router tests | Not applicable | IMPLEMENTED |
| Evidence, confidence and trace | Typed evidence, previews, rationales, uncalibrated disclosures and ordered safe trace | All production specialists | response/report/analytics tests | Task-specific only | IMPLEMENTED |
| Downloadable reports | Backend-generated PDF, JSON, CSV and ZIP | Report generator | report integration tests | Not applicable | IMPLEMENTED |
| Hidden Cartosat-2S/RISAT readiness | Sensor-agnostic numeric ingestion with conservative metadata handling | Input/SAR validators | dtype/band/metadata tests | Actual sensors not tested | EVIDENCE GAP |

## Representative regression queries

`tests/compliance/test_problem_statement_26167.py` preserves the five official representative query forms and asserts workflow validity, task classification, selected specialist family, permitted parameter surface, and explicit rejection of a directional change question submitted as a single-image workflow. Feature-specific endpoint suites validate non-null responses, evidence products, confidence/provenance, execution traces, and visible fallbacks without pinning exact answer wording.

## Evidence boundaries

- ChangerEx/deterministic mask overlap is consistency evidence, not ground-truth accuracy.
- SVE similarity is scene-level supporting evidence, not a probability or segmentation.
- Translated SAR is an optical-like learned representation, not optical ground truth.
- Non-georeferenced benchmark pairs may have assumed pixel alignment, but geospatial co-registration cannot be independently verified.
- Missing public datasets leave benchmark status unavailable or pending; SatQuery does not generate replacement scores.

### Optical + SAR preview repair

The official joint built-up/water query and “Where do both modalities agree?” support disclosed qualitative same-size PNG/JPEG pairs with validated source roles. Native optical and native SAR evidence remain distinct, and quantitative spatial fusion still requires exact verified grids. See `OPTICAL_SAR_ROOT_CAUSE_AUDIT.md` for the traced causes and verification; `OPTICAL_SAR_EVALUATION_PROTOCOL.md` defines the qualitative/quantitative boundary. No model artifacts or benchmark scores changed.
