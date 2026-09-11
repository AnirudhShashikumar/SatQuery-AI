# SatQuery AI frontend visual review

## Review scope

The redesigned assistant was exercised against the existing local backend using the approved offline demo samples. The review covered the empty workspace, single optical, single SAR, optical–SAR, bi-temporal, grounding, loading, unsupported-format error, and deterministic fallback states. Desktop, laptop, tablet, and mobile viewports were inspected.

## Findings

- **Hierarchy:** The page now leads with the product promise, then a persistent three-workflow selector, query composer, attachments, and results. The primary action remains visually dominant without hiding inputs.
- **Spacing and density:** The workspace uses a consistent card rhythm and compact metadata rows. Detailed provenance remains available without competing with the direct answer.
- **Typography:** Sentence-case labels, restrained uppercase eyebrows, and a clearer heading scale make long scientific responses easier to scan.
- **Contrast and state:** Semantic colors distinguish optical, SAR, change, grounding, confidence, readiness, warning, and fallback states. Focus and hover states remain explicit in light and dark themes.
- **Consistency:** Uploads, health, evidence participation, errors, and workflow selection share the same card, badge, and status vocabulary.
- **Scientific readability:** Direct answers, confidence, runtime, specialist, task, effective modality, representation, evidence, limitations, and provenance remain source-labelled. Generated previews and deterministic fallback are disclosed rather than presented as authoritative learned output.
- **Responsiveness:** The fixed desktop navigation collapses to the existing compact mobile treatment. The workflow cards, upload grid, result actions, and evidence panels stack without horizontal overflow at 1024 px and 390 px.
- **Evidence traceability:** Evidence participation is summarized at source level from backend-returned fields and execution traces. Low-level trace events are not repeated as separate evidence cards.
- **Warnings and recovery:** Unsupported files receive a concrete recovery action. Loading shows the active stage and elapsed time, with no fabricated percentage estimate.

## Captured states

- `report_assets/assistant-empty-desktop.png`
- `report_assets/assistant-mobile.png`
- `report_assets/single-optical-input-desktop.png`
- `report_assets/single-optical-result-desktop.png`
- `report_assets/single-optical-result-tablet.png`
- `report_assets/single-optical-result-mobile.png`
- `report_assets/single-sar-input-desktop.png`
- `report_assets/single-sar-result-desktop.png`
- `report_assets/cross-modal-input-desktop.png`
- `report_assets/cross-modal-result-desktop.png`
- `report_assets/bitemporal-input-desktop.png`
- `report_assets/bitemporal-result-desktop.png`
- `report_assets/bitemporal-result-laptop.png`
- `report_assets/grounding-result-desktop.png`
- `report_assets/analysis-loading-desktop.png`
- `report_assets/unsupported-format-error-desktop.png`
- `report_assets/fallback-state-desktop.png`

## Remaining limitations

- Model readiness is reported exactly as returned by the existing health endpoint; the frontend does not reinterpret an unloaded specialist as ready.
- A translated SAR representation is shown only when the backend returns one. Native-only SAR results intentionally omit a synthetic translated view.
- Fallback answers remain visibly labelled when learned evidence is unavailable.
- Browser review validates representative workflows and responsive states, but is not a substitute for a formal external WCAG certification.

No backend endpoint, inference implementation, request contract, response schema, analytics behavior, or model artifact was changed for this redesign.
