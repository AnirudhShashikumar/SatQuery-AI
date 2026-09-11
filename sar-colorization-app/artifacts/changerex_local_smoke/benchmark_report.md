# Standalone ChangerEx real-pair smoke report

> The masks and overlays are model predictions, not ground truth. This is an operational smoke and device/reuse benchmark only; it does not measure change-detection accuracy.

## Source pair

- Hanford, Washington, Landsat 7 ETM+, before May 6, 2000 and after July 9, 2000.
- Source: [NASA Earth Observatory](https://science.nasa.gov/earth/earth-observatory/high-resolution-view-of-hanford-washington-833/).
- Credit: Image courtesy Ron Beck, USGS EROS Data Center.
- Original dimensions: 2400 × 1801; both images are preserved in `input/` with SHA-256 values in `source_manifest.json`.

## Final smoke results

| Measure | MPS | CPU |
|---|---:|---:|
| Strict load | pass | pass |
| Device inference | pass | pass |
| Model load time | 0.2490 s | 0.1019 s |
| First end-to-end inference | 0.7625 s | 0.5580 s |
| First model forward | 0.3929 s | 0.3677 s |
| Warm model mean (5 runs) | 0.0960 s | 0.3164 s |
| Warm model median | 0.0952 s | 0.3139 s |
| Warm model P95 | 0.0981 s | 0.3245 s |
| Peak measured memory | 1121.0 MB MPS driver allocation | 1126.7 MB process RSS |
| Reuse count | 6 | 6 |
| Repeat max probability difference | 0.0 | 0.0 |
| Output probability range | 1.2882e-17–0.9847781 | 1.2883e-17–0.9847782 |
| Changed percentage | 0.163682% | 0.163682% |
| Connected regions | 5 | 5 |
| Output dimensions | 2400 × 1801 | 2400 × 1801 |

MPS and CPU produced identical binary masks. Their maximum absolute probability difference was `2.95043e-06`, with mean absolute difference `5.39955e-09`; this passes the package's float32 parity tolerances.

## Official-source parity

The extracted CPU implementation was compared with a separately instantiated reference that executed the unmodified pinned Open-CD Changer/interaction modules and MMSeg ResNet source. Only an inference-only framework import shim was supplied to avoid eager loading of unrelated MMCV custom operations.

- Preprocessed tensor: `[1, 6, 768, 1024]`
- Preprocessed tensor SHA-256: `4891c980f62096fa1b8ee4c95f4be3090b90ce33de7aec835a76453f369b7311`
- Logits: `[1, 2, 768, 1024]`
- Logits maximum/mean absolute difference: `0.0` / `0.0`
- Restored probability maximum/mean absolute difference: `0.0` / `0.0`
- Binary mask agreement: `1.0`
- Mask IoU: `1.0`
- Changed-percentage difference: `0.0` percentage points
- Acceptance: pass

See `parity/parity_report.json` and `parity/official_reference_metadata.json` for machine-readable evidence.
