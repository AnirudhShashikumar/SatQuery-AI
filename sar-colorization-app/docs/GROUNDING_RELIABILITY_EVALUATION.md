# Local Grounding Reliability Evaluation

GeoVision includes an offline evaluation framework for measuring the existing Grounding DINO checkpoint and operational reliability gate against locally obtained, manually labelled remote-sensing imagery.

This is an exploratory validation tool, not a bundled benchmark. The repository does not include copyrighted benchmark images or claim benchmark performance.

## Dataset layout

Create a directory outside the repository or in a locally ignored workspace:

```text
grounding_eval/
  annotations.json
  images/
    stadium_001.png
    water_negative_001.tif
```

`annotations.json` uses this contract:

```json
{
  "samples": [
    {
      "id": "stadium_001",
      "image": "images/stadium_001.png",
      "target": "stadium",
      "query": "Locate the stadium.",
      "target_present": true,
      "boxes": [[120, 80, 310, 245]],
      "source": "manual",
      "notes": null
    },
    {
      "id": "water_negative_001",
      "image": "images/water_negative_001.tif",
      "target": "water body",
      "query": "Highlight the water body.",
      "target_present": false,
      "boxes": [],
      "source": "manual",
      "notes": "Reviewer confirmed the target is absent."
    }
  ]
}
```

Rules:

- Sample IDs contain only letters, numbers, underscores, and hyphens.
- Image names must be relative to the dataset directory. Absolute paths and directory traversal are rejected.
- Images must exist and decode as PNG, JPEG, TIFF, or GeoTIFF-compatible TIFF.
- Positive samples require one or more valid `[x1, y1, x2, y2]` reference boxes.
- Negative samples must explicitly set `target_present` to `false` and use an empty boxes array.
- Boxes must have finite coordinates, positive area, and remain within the source-image bounds.
- Annotation source is required and should be a short label such as `manual` or `two-reviewer-consensus`.
- Missing annotations are rejected; the evaluator never invents boxes or target presence.

The initial analysis categories are stadium, aircraft, building, bridge, road, water body, river, lake, ship, and vegetation. Other non-empty target labels are retained and evaluated without crashing, although the production localized-area policy applies only to its configured localized-target list.

## Preparing local samples

Use imagery you are legally permitted to evaluate. Create reference boxes with a local annotation tool and retain the annotation provenance in `source`. Include positive and negative scenes, varied resolutions and target sizes, and urban, rural, coastal, and mixed settings.

Five to ten images per target is a practical minimum for local debugging, but such a collection is still too small for a statistically significant scientific claim. Review annotation consistency before interpreting model results.

Do not commit restricted imagery, credentials, private paths, or licensed benchmark data to the repository.

## Running the evaluator

```bash
python scripts/evaluate_grounding_reliability.py \
  --dataset ./grounding_eval \
  --output ./artifacts/grounding_eval \
  --device auto \
  --reuse-model
```

Optional analysis filters and overrides:

```text
--device auto|cpu|mps|cuda
--limit N
--target "water body"
--minimum-score 0.45
--maximum-area-ratio 0.85
--reuse-model
--no-reuse-model
```

The defaults reference the production reliability gate: minimum alignment score `0.45` and maximum localized-target area ratio `0.85`. CLI overrides are analysis-only. They do not mutate the production defaults or automatically select a replacement threshold.

The evaluator collects candidates down to score `0.30` solely so the declared threshold-analysis grid can be computed. Grounding DINO inference weights, prompts, processor resizing, and coordinate restoration remain unchanged.

## Outputs

The output directory contains:

- `summary.json`
- `per_sample.csv`
- `per_target.csv`
- `rejected_candidates.csv`
- `false_localizations.csv`
- `threshold_grid.csv`
- `evaluation_report.md`

Exports contain safe sample IDs and annotation labels, not image filenames, absolute paths, image bytes, environment variables, or model-cache locations. Source images and annotations are read-only.

Metrics distinguish model candidates, gate decisions, and localization accuracy. They include acceptance/rejection counts and rates, one-to-one precision and recall at IoU 0.25 and 0.50, false positives, false localizations, missed targets, correct negative rejections, score and area-ratio distributions, mean and median accepted best IoU, center inclusion, and full-frame candidate rates. Metrics without a valid denominator are `null`/`Unavailable` rather than fabricated as zero.

The threshold grid evaluates score values `0.30–0.55` and area-ratio values `0.60–0.95`. It never changes production settings and does not recommend a threshold automatically.
