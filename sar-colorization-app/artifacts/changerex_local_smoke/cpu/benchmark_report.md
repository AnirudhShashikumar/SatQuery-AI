# ChangerEx standalone benchmark report

> The binary mask and overlay are model predictions, not ground truth. This operational smoke run does not measure accuracy.

Input: NASA Earth Observatory/USGS Hanford Landsat pair documented in `../source_manifest.json`.

## Input and output

- Source dimensions: 2400 × 1801
- Model input dimensions: 1024 × 768
- Device: cpu
- Threshold: 0.5
- Probability range: 0.00000000–0.98477823
- Changed percentage: 0.163682%
- Connected regions: 5

## Runtime

- Model load: 0.10186591700000003 s
- First inference: 0.557952542 s
- First model forward: 0.36772641699999997 s
- Benchmark runs: 5
- Mean: 0.3164361080000001 s
- Median: 0.313925416 s
- P95: 0.3245433750000001 s
- Peak memory: 1126.71875 MB
- Reuse count: 6
- Deterministic repeatability: True

## Limitations

- The output is a model prediction, not ground truth.
- LEVIR-CD primarily represents building change and may not generalize to all land-cover changes.
- Inputs must be geometrically co-registered; misregistration can appear as change.
