# SatQuery Grounding Specialist v1 Pilot — VRSBench Smoke-100

This evaluation uses the same 100 records as the unchanged Grounding DINO baseline. The base `IDEA-Research/grounding-dino-tiny` model was frozen, the exported specialist state was loaded strictly, and every complete referring sentence was passed to the base processor.

The deployed specialist prediction is the query with the highest specialist logit. Ground-truth IoU was computed only after selection. No best-IoU oracle selection, production reliability gate, box threshold, or text threshold was applied.

## Results

| Metric | Specialist | Baseline | Change |
|---|---:|---:|---:|
| Mean IoU | 0.2644 | 0.1283 | +0.1361 |
| Accuracy@0.25 | 35.00% | 22.00% | +13.00% |
| Accuracy@0.50 | 28.00% | 13.00% | +15.00% |
| Accuracy@0.75 | 18.00% | 6.00% | +12.00% |
| Null/failure rate | 0.00% | 64.00% | -64.00% |
| Average latency | 1231.4 ms | 1169.1 ms | +62.2 ms |

The baseline used production filtering and could return null. The specialist always selects one learned query unless inference fails; its lower null rate is therefore not directly equivalent to better localization.

## Class and clipping analysis

Best five by accuracy@0.50: helipad (100%), stadium (75%), basketball-court (75%), expressway-service-area (75%), helicopter (75%).

Worst five by accuracy@0.50: vehicle (0%), roundabout (0%), harbor (0%), ground-track-field (0%), trainstation (0%).

Clipped accuracy@0.50: **35.00%**; unclipped: **23.33%**.

## Query and geometry analysis

The specialist selected **28** unique query indices. Top-1 / top-5 / top-10 coverage is **49.00% / 73.00% / 82.00%**. Collapse detected: **True** (one_query_index_above_25_percent, top_five_query_indices_above_60_percent).

Average predicted / ground-truth normalized box area is **0.3350 / 0.1116**. Oversized / undersized rates are **46.00% / 12.00%**.

## Recommendation

**B. continue to 1,000-step balanced training** under the transparent evaluation rule encoded in this script.

## Reproducibility

```bash
venv/bin/python scripts/run_vrsbench_grounding_specialist_smoke.py --checkpoint /Users/anirudhshashikumar/Desktop/VRSBench/pilot_step_100.pt --manifest /Users/anirudhshashikumar/Desktop/VRSBench/grounding_smoke100/vrsbench_grounding_smoke100.jsonl --image-root /Users/anirudhshashikumar/Desktop/VRSBench/grounding_smoke100/images --output-dir '/Users/anirudhshashikumar/Documents/Projects/SatQuery AI/sar-colorization-app/artifacts/vrsbench_grounding_smoke_specialist_step100' --device cpu --batch-size 1
```

This Smoke-100 pilot does not establish full-dataset generalization or calibrated confidence. No training, threshold tuning, checkpoint change, or production behavior modification occurred.
