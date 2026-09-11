# Official RSVQA-LR Test Benchmark — SatQuery RSVQA Specialist v1

## Scope and reproducibility

This measurement uses the official RSVQA-LR test split: **10,004 questions across 100 images**. The model is **SatQuery RSVQA Specialist v1**. No training, routing, inference, vocabulary, preprocessing, checkpoint, or benchmark label was changed for this run.

Architecture: OpenCLIP ViT-L-14 (`laion2b_s32b_b82k`) with the frozen SatQuery Vision Encoder v1 adapter, projected image/question features, elementwise product and absolute-difference fusion, then exported presence, comparison, rural/urban, and count heads.

Checkpoint SHA-256: `71c0ab56ee650813bd495e8a3bc777353b6907a097af860e417f60523efe56ad`.

Preprocessing reads the first three TIFF bands, applies the verified per-band 2nd–98th percentile stretch, clips to [0,1], converts to uint8 RGB PNG, and applies the OpenCLIP ViT-L-14 validation transform. Converted images are cached by image ID.

Exact match compares normalized yes/no, rural/urban, non-negative integer strings, and the literal `201+` class. Official labels above 200 remain their exact integer values. Overflow-aware accuracy is reported separately by mapping official values above 200 to the exported `201+` policy.

Reproduction command:

```bash
venv/bin/python scripts/run_rsvqa_full.py --manifest '/Users/anirudhshashikumar/Documents/Projects/SatQuery AI/sar-colorization-app/artifacts/rsvqa_full_specialist_v1/rsvqa_lr_test_manifest.jsonl' --image-root /Users/anirudhshashikumar/Desktop/RSVQA/images/Images_LR --endpoint http://127.0.0.1:8010/api/analysis/image --output-dir '/Users/anirudhshashikumar/Documents/Projects/SatQuery AI/sar-colorization-app/artifacts/rsvqa_full_specialist_v1' --timeout 120 --checkpoint-every 25 --max-retries 3
```

## Test results

| Metric | Result |
|---|---:|
| Exact-match accuracy | 71.05% |
| Overflow-aware accuracy | 74.31% |
| Macro task accuracy | 74.05% |
| Coverage | 100.00% |
| Null answers | 0 |
| Endpoint errors | 0 |
| Specialist-use rate | 100.00% |
| Fallback-use rate | 0.00% |

| Task | Samples | Exact accuracy | Coverage | Majority baseline | Exceeds baseline |
|---|---:|---:|---:|---:|:---:|
| presence | 2,955 | 91.40% | 100.00% | yes (75.03%) | yes |
| comp | 4,002 | 88.48% | 100.00% | no (66.74%) | yes |
| count | 2,947 | 26.33% | 100.00% | 0 (25.55%) | yes |
| rural_urban | 100 | 90.00% | 100.00% | rural (56.00%) | yes |

## Latency

| Metric | Value |
|---|---:|
| Average | 26.9 ms |
| Median | 29.0 ms |
| P90 | 29.0 ms |
| P95 | 30.0 ms |
| P99 | 54.0 ms |
| Throughput | 31.65 questions/s |
| Wall-clock duration | 316.1 s |

## Count analysis

Exact count accuracy is **26.33%** and overflow-aware count accuracy is **37.39%**. Numeric-only MAE is **14.23758719737382** across 2,437 pairs; `201+` is excluded from numeric error. Within ±1 / ±2 / ±5 is **43.46% / 51.01% / 62.99%**.

Counts are learned RSVQA benchmark labels. They are not physical object inventories, segmentation measurements, or grounding results.

## Validation and prior smoke references

Exported model-card results below are validation metrics, not test results:

| Task | Exported validation | Official test |
|---|---:|---:|
| presence | 90.85% | 91.40% |
| comp | 91.00% | 88.48% |
| count | 37.56% | 26.33% |
| rural_urban | 90.00% | 90.00% |

The earlier heuristic and specialist Smoke-50 runs are small-set references only and are not substitutes for this full-test measurement.

| Smoke-50 system | Accuracy | Coverage | Average latency |
|---|---:|---:|---:|
| Old heuristic | 38.00% | 70.00% | 1060.3 ms |
| RSVQA Specialist v1 | 80.00% | 100.00% | 164.0 ms |

These rows compare the two systems on the same 50-question smoke manifest. The official 10,004-question result above must not be directly subtracted from the heuristic Smoke-50 accuracy.

## Error analysis

Incorrect/error records: **2,896**. High-confidence errors (confidence ≥0.90): **523**. The highest-confidence error category is **count**.

Among count failures, the most common prediction is **201+** (23.49%); count-class collapse flag: **False**.

Detailed wording, entity, image, confidence, count-bucket, and fallback groupings are saved in the CSV and JSON artifacts. Analysis was performed only after predictions were produced.

## Fallback behavior and limitations

The specialist answered 100.00% of all records; fallback use was 0.00%. Endpoint failures are retained as explicit rows and never converted into model predictions.

Confidence values are task-head softmax scores and are not calibrated probabilities. This benchmark does not establish physical object-count accuracy, segmentation accuracy, grounding accuracy, or generalization beyond RSVQA-LR. Results apply to the saved official split, deterministic rendering pipeline, checkpoint, and runtime configuration.
