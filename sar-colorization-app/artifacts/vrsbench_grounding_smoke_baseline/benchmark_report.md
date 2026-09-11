# VRSBench Grounding Smoke-100 — SatQuery Grounding DINO baseline

This is a measurement-only run of the unchanged `IDEA-Research/grounding-dino-tiny` checkpoint. Full VRSBench referring sentences were passed to the repository's existing in-process candidate provider. The current processor thresholds, reliability gate, NMS, and detection cap were retained.

Thresholds: box **0.35**, text **0.25**, reliability score **0.45**, NMS IoU **0.5**. Model loads: **1**; reuses including preflight: **104**.

## Results

| Metric | Value |
|---|---:|
| Completed | 100/100 |
| Mean IoU, null=0 | 0.1283 |
| Non-null mean IoU | 0.3563478489472691 |
| Accuracy@0.25 | 22.00% |
| Accuracy@0.50 | 13.00% |
| Accuracy@0.75 | 6.00% |
| Null rate | 64.00% |
| Average / P95 latency | 1169.1 / 1201.3 ms |
| Runtime | 117.1 s |

The benchmark prediction is always the accepted box with highest ground-truth IoU, not necessarily the highest-confidence box. This oracle matching is evaluation-only.

## Class extremes

Best five by accuracy@0.50: helicopter (75%), stadium (50%), basketball-court (50%), swimming-pool (25%), expressway-toll-station (25%).

Worst five by accuracy@0.50: airplane (0%), container-crane (0%), dam (0%), helipad (0%), tennis-court (0%).

## Clipping

Clipped accuracy@0.50: **12.50%**; unclipped: **13.33%**. Detailed mean IoU and null rates are in `clipped_vs_unclipped.csv`.

## Failure taxonomy

Categories use only returned labels/scores, box geometry, clipping metadata, uniqueness, and explicit relation metadata. They are diagnostic buckets, not claims about hidden model cognition.

## Recommendation

**C. fine-tune/replace grounding specialist** under the documented rule encoded in the evaluator. Threshold tuning cannot be claimed to improve results until a separate, explicitly authorized threshold sweep is run.

## Limitations and reproducibility

Scores are text-region alignment scores, not calibrated confidence. Smoke-100 is small and does not establish general grounding performance. Best-IoU matching is optimistic when several boxes are returned. Clipped annotations can constrain achievable overlap. No training, threshold change, or production behavior modification occurred.

```bash
venv/bin/python scripts/run_vrsbench_grounding_smoke.py --manifest /Users/anirudhshashikumar/Desktop/VRSBench/grounding_smoke100/vrsbench_grounding_smoke100.jsonl --image-root /Users/anirudhshashikumar/Desktop/VRSBench/grounding_smoke100/images --endpoint in-process://satquery/rs_grounder --output-dir '/Users/anirudhshashikumar/Documents/Projects/SatQuery AI/sar-colorization-app/artifacts/vrsbench_grounding_smoke_baseline' --timeout 120
```
