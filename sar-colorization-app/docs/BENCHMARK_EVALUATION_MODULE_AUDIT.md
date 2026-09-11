# SatQuery AI Benchmark & Evaluation Module Audit

Audit date: 2026-09-01  
Source commit before implementation: `c251991aa696e4dcd37ddd926a6c2be17042a096`

## Scope and non-regression boundary

The audit covered `models/`, `artifacts/`, `docs/`, `scripts/`, `tests/`, `frontend/`, `satquery_agent/`, `changerex_local/`, and `grounding_eval/`. The benchmark module is a read-only reporting layer over versioned JSON. It does not call inference, alter checkpoints, change preprocessing, or modify backend routes, request/response models, routing, analytics, grounding, RSVQA, captioning, SVE, SAR translation, ChangerEx, deterministic change analysis, cross-modal fusion, or existing analysis report generation.

## Existing result inventory

| Area | Source artifact | Evidence class | Dataset / split | Samples | Disposition |
|---|---|---:|---|---:|---|
| RSVQA Specialist v1 | `artifacts/rsvqa_full_specialist_v1/results.json` plus run state and manifest provenance | Verified test | RSVQA-LR official test | 10,004 questions / 100 images | Import as preferred `verified_test` |
| RSVQA Specialist v1 | `models/rsvqa_specialist_v1/validation_metrics.json` | Verified validation | RSVQA-LR official validation | 10,005 | Import as historical `verified_validation` |
| RSVQA smoke variants | `artifacts/rsvqa_smoke*/results.json` | Smoke only | 50-question subsets | 50 each | Do not promote; superseded by complete official test result |
| Grounding Specialist v1.1 | `models/grounding_specialist_v1_1/validation_results.json` | Verified validation | VRSBench `grounding_v2` validation | 16,146 | Import as preferred `verified_validation` |
| Grounding DINO baseline | `artifacts/vrsbench_grounding_smoke_baseline/results.json` | Smoke only | Prepared VRSBench Grounding Smoke-100 | 100 | Import as historical `smoke_only` |
| Grounding pilot step 100 | `artifacts/vrsbench_grounding_smoke_specialist_step100/results.json` | Smoke only | Same Smoke-100 subset | 100 | Documented but not imported as preferred; checkpoint differs from production step 600 |
| Grounding score modes | `artifacts/grounding_score_mode_comparison/results.json` | Operational only | One local image / four score modes | 1 | Skip as quality evidence; useful regression diagnostic only |
| ChangerEx | `artifacts/changerex_local_smoke/mps/result.json`, parity and environment files | Operational only | NASA/USGS Hanford pair, no reference mask | 1 pair | Import runtime/repeatability only; no accuracy metrics |
| ChangerEx production smoke | `artifacts/changerex_production_smoke/smoke_summary.json` | Operational only | Same real pair | 1 pair | Skip as duplicate operational evidence; mask agreement is not accuracy |
| Pix2Pix SAR translation | `artifacts/single_sar_translation_smoke/comparison.json` | Operational only | Repository single-SAR smoke input, no paired target | 3 requests | Import runtime only |
| SARFusionFormer | single-image workflow smoke files | Operational only | No paired target | small smoke | Represent as benchmark unavailable; no quality metrics |
| Color Corrector | production pipeline metadata | Component only | — | — | Not a standalone benchmark |
| Captioner | caption smoke behavior in tests/runtime | Operational integration only | No quality benchmark artifact | — | Benchmark unavailable |
| Optical + SAR fusion | cross-modal tests and SVE runtime validation | Operational integration only | No ground-truthed controlled protocol | — | Benchmark unavailable |
| SatQuery Vision Encoder | `models/satquery_vision_encoder_v1/baseline_vs_adapted.json` | Verified held-out retrieval evaluation | BigEarthNet-derived test data | 2,053 pairs | Audited but outside the six requested benchmark cards |
| TTP | `artifacts/ttp_production_smoke/smoke_results.json` | Failed operational smoke | — | 1 | Excluded; ChangerEx is the default learned change engine and TTP remains alternate |

## Scientifically valid sources

The strongest repository result is the complete RSVQA-LR official test run. Its run state records all 10,004 official questions, no endpoint errors, no task hints sent to the model, no ground truth sent to the endpoint, and the checkpoint SHA-256. The Grounding Specialist result is a full known VRSBench validation run with 16,146 records and a verified bundle fingerprint. It is correctly labeled validation, never test.

The RSVQA epoch-8 file is valid validation evidence but is secondary to the official test run. The Grounding Smoke-100 artifacts are useful for regression and failure analysis, but their curated subset status prevents a full-benchmark claim.

## Operational-only evidence

ChangerEx MPS/CPU parity proves exact extraction and local execution. The Hanford pair has no supplied mask, so repeatability, latency, memory, and parity are operational metrics only. The Pix2Pix smoke proves routing and model reuse but has no paired optical target. SVE runtime values describe lifecycle performance, not caption, VQA, grounding, change, or fusion accuracy. Grounding score-mode outputs are one-image regression diagnostics. None are promoted to accuracy.

## Models and checkpoint fingerprints

- RSVQA Specialist v1: `rsvqa_specialist_v1_head.pt`, SHA-256 `71c0ab56ee650813bd495e8a3bc777353b6907a097af860e417f60523efe56ad`.
- Grounding Specialist v1.1 step 600: `grounding_specialist_v1_1_head.pt`, SHA-256 `5e8db30becadb1d063fc0154614ee2a2a4b7a2c2923db3fce4ff036c65007342`.
- ChangerEx ResNet-18: `ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth`, SHA-256 `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618`.
- Pix2Pix production checkpoint is named in runtime code, but no repository SHA artifact accompanies the smoke result; the benchmark record therefore leaves the fingerprint `null`.

## Runtime and memory evidence

- RSVQA official test: mean 26.874 ms, median 29 ms, P95 30 ms, 31.65 questions/s. Hardware/device were not captured and remain undisclosed.
- Grounding v1.1 validation: mean 34.829 ms, median 34.594 ms, P95 36.227 ms, peak 8.178 GiB. The artifact identifies GPU execution but not the GPU model.
- ChangerEx local MPS: five warm runs, mean 95.966 ms, P95 98.147 ms, peak process memory 1120.969 MB, Apple Silicon/macOS 26.6.2/PyTorch 2.8.0.
- Pix2Pix operational smoke: 80 ms translation runtime and 1004.52 MB peak process memory. This is local-machine context, not universal performance.

## Existing frontend and exports

The existing `/benchmark` navigation route rendered `frontend/components/benchmark-view.tsx`. It contained a hard-coded RSVQA Smoke-50 snapshot and illustrative Pix2Pix/SARFusionFormer numbers. The page warned that the SAR values were illustrative, but it had no strict data loader, historical provenance, compatible-comparison guard, or benchmark exports. Existing analysis report utilities support PDF/JSON/CSV/ZIP; they are intentionally not modified. The benchmark module receives a separate export implementation.

## Demo assets

The repository contains an attributed NASA/USGS Hanford before/after pair and three VRSBench grounding evaluation images with repository annotations. Only the Hanford pair and two well-described grounding cases are included initially. Hanford is classified `curated_expected_behavior`, not ground truth. The grounding cases retain their annotation provenance and launch the real workspace through guided upload instructions.

## Missing coverage

- Caption quality: no BLEU, METEOR, ROUGE-L, CIDEr, or SPICE run.
- ChangerEx quality: no SatQuery-reproduced LEVIR-CD test metrics.
- Pix2Pix/SARFusionFormer: no common paired test split with PSNR, SSIM, LPIPS, or L1.
- Optical + SAR fusion: no disclosed controlled protocol with ground truth or human-review adjudication.
- Hardware identity is absent from RSVQA and Grounding validation artifacts.
- Model licenses are incomplete for some repository-only checkpoints.

## Recommended schema and implementation

Use repository-level `benchmarks/` JSON with schema version `1.0.0`. Every record carries model identity, exact checkpoint/fingerprint when known, evaluation status, dataset, split, sample count, protocol, timestamp, environment, task-appropriate metric rows, performance, source artifacts, limitations, and reproduced/external provenance. Missing numbers are `null`; zero remains a real zero. A suite manifest selects the preferred record per specialist while preserving historical records.

The frontend should validate records at runtime, reject malformed inputs without crashing, select the preferred verified run, prevent cross-task quality comparisons, display status and protocol context beside every metric, and derive JSON/CSV/PDF exports directly from normalized records. The demo gallery should use the separate demo schema and guided links into existing workflows.

## Explicit implementation boundaries

Permitted changes are limited to benchmark/demo JSON, schemas, import/validation utilities, frontend reporting components/routes/styles, focused tests, and documentation. Production Python inference and backend APIs are read-only for this task. No benchmark runs are executed automatically or as part of page rendering.
