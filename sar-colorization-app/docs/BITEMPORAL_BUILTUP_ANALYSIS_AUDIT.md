# Bi-temporal built-up analysis audit

Date: 2026-09-06

## Limitation found

The existing semantic interpreter correctly refused to derive a land-cover transition from a binary mask, but it summarized mostly global changed-pixel percentage, the largest component, and optional scene-level SVE prior deltas. It did not compare optical structural evidence inside each changed component, did not publish directional region objects, and could not render numbered top-region evidence. Built-up answers were therefore safe but generic.

## Evidence pipeline

ChangerEx remains the primary learned detector. Its mask supplies geometry only. For up to the three largest meaningful components, the new local analyzer computes:

- image-relative location and the real component bounding box;
- component pixels, relative area, bounding-box fill compactness, and coarse change strength;
- before/after edge density, gradient strength, texture, and a vegetation-discounted built-up-like support value;
- overlap with the independent deterministic mask;
- a conservative per-region directional state and semantic support level.

The fixed minimum displayed region size is 0.02% of the analysis grid. Built-up direction requires a built-up-like support delta of at least 0.06 and support of at least 0.20 on the relevant side. These are documented application heuristics, not benchmark-standard thresholds or calibrated probabilities.

## Direction states

- `INCREASE_SUPPORTED`: at least one meaningful changed region has stronger later built-up-like structural support.
- `DECREASE_SUPPORTED`: the reverse comparison is supported.
- `MIXED_CHANGE`: supported increase and decrease regions both occur.
- `NO_MEANINGFUL_EVIDENCE`: no meaningful learned component is available.
- `INSUFFICIENT_EVIDENCE`: change exists but structural direction is not established.

The answer is composed as WHAT, WHERE, MAGNITUDE, and WHY. It uses north/south/east/west/center language without georeferencing and never converts pixels to physical area. Generic and directional answers continue to disclose that the interpretation is heuristic and not ground-truth land cover.

## Confidence

Low/moderate/high are descriptive, uncalibrated levels. Factors are published individually: ChangerEx geometry, deterministic regional overlap, before/after direction strength, and region coherence. Limited learned/deterministic agreement lowers the result. A binary mask, texture, brightness, edge density, or SAR return can never independently establish buildings.

## Visual and UI changes

- `Top Changed Regions` is generated from real connected-component boxes over the later observation.
- Outlines are numbered and described as changed regions with built-up-like evidence, never as a built-up segmentation mask.
- Judge View shows up to three region cards with location, structural strength, directional state, and relative size.
- Research View retains masks/statistics and now receives structured before/after support values, deterministic overlap, confidence factors, and limitations in the response schema.
- Execution trace adds region extraction, structural evidence extraction, built-up evidence comparison, semantic interpretation, confidence metadata, and response generation only when those stages execute.

## Real pair observation

The existing attributed Hanford pair completed through ChangerEx on CPU with no fallback. ChangerEx found eight components (0.527064% changed); the top three were analyzed and a numbered overlay was generated. The directional built-up query returned `MIXED_CHANGE`, while mask IoU with deterministic evidence was only 0.007389. The result therefore remains a cautious built-up-like structural interpretation and must not be treated as urban ground truth, especially because this pair depicts brushfire effects outside the model's primary building-change domain.

Manual browser verification on the isolated updated app completed both `Has the built-up area increased?` and `Where has urban expansion occurred?`. The latter exposed and then verified a repaired routing vocabulary gap: urban/built-up expansion language now enters the bi-temporal change path instead of returning `unsupported_task`. Both queries published the same evidence-derived northwest/center mixed-change interpretation, the three real region cards, ChangerEx provenance, overlap statistics, and limitations.

## Integrity

- ChangerEx checkpoint and pure-PyTorch architecture: unchanged.
- Protected SHA-256 remains `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618`.
- Deterministic analysis remains independent supporting evidence/fallback.
- No semantic class is derived from the binary mask alone.
- No geographic area or coordinates are fabricated.
- No hardcoded final answer strings tied to particular imagery were added.
