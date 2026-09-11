# Benchmark Result Migration Report

Generated: 2026-09-01T14:15:53Z

## Imported records

| Benchmark ID | Status | Source | Reason |
|---|---|---|---|
| `rsvqa.specialist-v1.official-test.2026-08-29` | `verified_test` | RSVQA full results, run state, model card, SHA manifest | Complete official test manifest with validation gates |
| `rsvqa.specialist-v1.validation.epoch8` | `verified_validation` | Exported validation metrics and model card | Known official validation split |
| `grounding.specialist-v1-1.vrsbench-validation.step600` | `verified_validation` | Grounding bundle validation result/model card/SHA | Full VRSBench validation artifact for production checkpoint |
| `grounding.dino-tiny.vrsbench-smoke100.baseline` | `smoke_only` | Baseline Smoke-100 results and run state | Retained for historical regression analysis, never full accuracy |
| `changerex.r18.hanford-mps-operational` | `operational_only` | MPS result, environment, parity, source manifest | Runtime and parity evidence without a ground-truth mask |
| `sar-translation.pix2pix.single-image-operational` | `operational_only` | Single-SAR comparison and report | Runtime/model reuse only; no paired target |
| `captioning.unavailable.v1` | `unavailable` | Repository audit | No quality benchmark found |
| `sar_translation_sarfusionformer.unavailable.v1` | `unavailable` | Repository audit | No valid paired quality result found |
| `cross_modal.unavailable.v1` | `unavailable` | Repository audit | No controlled ground-truthed fusion evaluation found |

All numerical values are copied without rounding into canonical JSON. Absolute source paths, usernames, local dataset paths, and commands containing them are excluded from display records.

## Skipped or not promoted

| Artifact family | Reason |
|---|---|
| RSVQA Smoke-50 and variant comparisons | Superseded by complete official test; smoke must not be presented as benchmark accuracy |
| Grounding Specialist pilot step 100 | Non-production checkpoint and Smoke-100 subset; retained in source artifacts only |
| Grounding score-mode comparison | One-image production regression diagnostic, not a quality benchmark |
| ChangerEx production smoke | Duplicates operational evidence; learned/deterministic mask agreement is explicitly not accuracy |
| SARFusionFormer workflow smoke | Execution without paired ground truth; represented as unavailable quality benchmark |
| SVE runtime validation | Lifecycle/supporting-evidence validation, not quality evidence for the six requested task cards |
| TTP failed smoke | Failed operational alternate-engine run; not relevant quality evidence for default ChangerEx |
| Existing hard-coded frontend demo values | Illustrative values are removed from the evidence dashboard and never migrated |

## Assigned status rationale

`verified_test` is reserved for the complete official RSVQA-LR test split. `verified_validation` is used only for named, known validation splits. Smoke subsets remain `smoke_only`. Local latency, repeatability, parity, reuse, and health evidence are `operational_only`. No external paper metrics are imported, so `external_reported` coverage remains zero. Specialists without defensible quality evidence receive explicit `unavailable` records.

## Missing coverage

The suite still needs a reproduced ChangerEx quality benchmark, a common paired SAR translation test, a standard caption benchmark, and a disclosed controlled cross-modal fusion protocol. Hardware metadata should be captured in future RSVQA and grounding runs.
