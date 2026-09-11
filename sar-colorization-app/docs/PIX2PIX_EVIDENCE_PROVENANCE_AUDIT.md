# Pix2Pix Evidence Provenance Consistency Audit

## Scope

This repair is limited to the optional Single Image SAR translation branch. It does not change Pix2Pix inference, checkpoints, benchmark values, the native SAR-primary evidence hierarchy, Optical + SAR, bi-temporal analysis, or Grounding.

## Root cause

Pix2Pix generation and downstream optical-like semantic interpretation were represented by one coarse `status` field. A successfully generated preview could therefore coexist with `completed_with_limitations`, while the frontend source card treated every value other than literal `success` as `FAILED`. The viewer separately trusted `generated_preview_url`, so the two surfaces disagreed. Recursive report preview discovery could also export a loose legacy preview field without validating its run or source identity.

## Authoritative lifecycle

The backend now records `NOT_REQUESTED`, `NOT_ELIGIBLE`, `RUNNING`, `SUCCEEDED`, or `FAILED` independently for generation and translation semantic comparison. Only `SUCCEEDED` generation may publish a `generated_optical_like` evidence product. That product carries an evidence ID, result/run ID, source observation ID, source SAR modality, generator, lifecycle status, type, disclosure, and preview reference.

Publication occurs only after inference output validation and successful publication of the principal generated artifact. Optional normalized or color-corrected artifact failures are separate stages. A semantic specialist failure does not relabel successful Pix2Pix generation; it is recorded as `translation_semantic_comparison = FAILED`, and its absent/failed claim is excluded from fusion and answer synthesis.

## Failure and legacy behavior

When generation fails, native SAR evidence remains available, the response is `COMPLETED_WITH_LIMITATIONS`, the Pix2Pix source is failed, and no generated product, translated grounding product, blend/split layer, or translation-derived agreement is published. Unexpected runtime exceptions follow the same safe contract and are not exposed verbatim.

Cached products must match the current result ID and primary source observation ID. The cache tool version was advanced to `satquery-sar-routing-1.2-pix2pix-evidence`, preventing reuse of older cache entries. Stored or legacy records without sufficient provenance are not upgraded to success: generated fields and semantic claims are suppressed with `Legacy generated evidence provenance unavailable.`

## Viewer and exports

The source card and viewer resolve the same authoritative product. A successful generated product is `USED`; successful generation without a publishable product is `AVAILABLE`; an ineligible branch is `SKIPPED`; and failed generation is `FAILED`. Agreement is rendered only when the separate semantic comparison succeeded.

JSON reports retain the authoritative lifecycle and product metadata. PDF evidence pages use only validated report preview products. ZIP packages include the same images plus `evidence/manifest.json`; failed or stale Pix2Pix products are absent from all three formats. Stored comparison/presentation previews use the same run/source validation.

## Verification matrix

- Successful generation: run-scoped product, `USED` source, disclosure, layer/blend/split, cache reuse.
- Inference failure: `FAILED`, no generated product, native SAR retained, completed with limitations.
- Semantic comparison failure: generation remains `SUCCEEDED`; semantic stage is failed; no failed claim or agreement banner is used.
- Stale/legacy product: mismatched run identity is suppressed.
- Reports: JSON, PDF preview mapping, ZIP image set, and ZIP evidence manifest agree.
- Frontend: source card and evidence viewer are tested from the same explicit metadata.

The focused lifecycle, SAR translation, API, cache, report, comparison, and frontend tests exercise these contracts. Full-suite and browser verification results are recorded in the completion report for this task.
