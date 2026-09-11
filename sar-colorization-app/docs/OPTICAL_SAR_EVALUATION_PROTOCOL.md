# Optical + SAR Controlled Evaluation Protocol

SatQuery does not report a generic “fusion accuracy.” Evaluation uses the same labelled samples and same task metric for three paths: `OPTICAL_ONLY`, `SAR_ONLY`, and `OPTICAL_PLUS_SAR`.

Supported task strata are water presence, built-up presence, object/region localization, cross-modal agreement, and complementary evidence. Every JSONL manifest record must identify the task, optical image, SAR image, exact question, and—when quantitative correctness is intended—the reference answer or localization reference. The evaluator records answers, participating specialists, fallbacks, and evidence origins for every path.

Fusion gain is calculated only when every compared sample has the same reference and exact-answer metric:

`fused score - max(optical-only score, SAR-only score)`

Localization must use an IoU-based adapter and must not be mixed with exact-answer scores. Generated optical-like SAR is supporting evidence only and must retain the disclosure “Learned optical-like representation; not optical ground truth.” Native SAR evidence cannot be replaced by translation.

Run:

```bash
python3 scripts/evaluate_optical_sar.py \
  --manifest /path/to/controlled.jsonl \
  --dataset-root /path/to/images \
  --output-dir artifacts/optical_sar_controlled
```

No suitable labelled paired dataset is bundled. Until an approved dataset and references are supplied, the canonical benchmark remains `unavailable`; operational demo completion is not accuracy evidence.

## Preview-pair operational checks

Same-size PNG/JPEG pairs with identifiable Optical and SAR content can complete qualitative queries without georeferencing. The disclosure is: “Pixel alignment is assumed from the supplied pairing; geospatial co-registration cannot be independently verified.” Independent candidate evidence is compared in nine image-relative sectors. Sector co-occurrence is not pixel-level overlap, geographic direction, semantic certainty, or benchmark accuracy. No joint masks, percentages, coordinates or metric areas are emitted.

Only `EXACT_GRID_MATCH` currently enables quantitative joint fusion. Shifted grids, different CRS and insufficient overlap retain their existing classifications and require explicit external correction; this repair adds no registration or resampling. Unreferenced TIFFs and different-size previews retain the protected unavailable/alignment behavior. Quantitative preview questions return partial/unavailable while supported qualitative questions return `success` / `COMPLETED`.

Slot roles and detected content are validated separately. Reversed or ambiguous slots block analysis; the UI offers a user-clicked swap for reversed files and re-inspects both. Native optical and SAR products retain observation IDs/roles/modality in results and exports. Optional translation is not invoked by this native-pair baseline; any translation shown elsewhere remains supporting only.
