# ChangerEx standalone benchmark report

> The binary mask and overlay are model predictions, not ground truth. This operational smoke run does not measure accuracy.

Input: NASA Earth Observatory/USGS Hanford Landsat pair documented in `../source_manifest.json`.

## Input and output

- Source dimensions: 2400 × 1801
- Model input dimensions: 1024 × 768
- Device: mps
- Threshold: 0.5
- Probability range: 0.00000000–0.98477811
- Changed percentage: 0.163682%
- Connected regions: 5

## Runtime

- Model load: 0.248962209 s
- First inference: 0.7624652080000001 s
- First model forward: 0.392881583 s
- Benchmark runs: 5
- Mean: 0.0959663168 s
- Median: 0.095201584 s
- P95: 0.09814729999999994 s
- Peak memory: 1120.96875 MB
- Reuse count: 6
- Deterministic repeatability: True

## Limitations

- The output is a model prediction, not ground truth.
- LEVIR-CD primarily represents building change and may not generalize to all land-cover changes.
- Inputs must be geometrically co-registered; misregistration can appear as change.
