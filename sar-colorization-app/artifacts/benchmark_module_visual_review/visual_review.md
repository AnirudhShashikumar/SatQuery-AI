# Benchmark Module Visual Review

Reviewed locally against the production Next.js application on 2026-09-02. This is a presentation and interaction review only; it did not invoke or modify inference.

## Routes and states reviewed

- `/benchmark`: overview, specialist cards, unavailable-evidence state, comparison guard, performance comparison, and JSON/CSV/PDF export controls.
- `/demos`: gallery, attribution, evidence labels, guided workflow links, and expanded demo detail.
- Viewports: 1440 px, 1280 px, 1024 px, and 390 px wide.
- Responsive result: no horizontal overflow at any reviewed breakpoint.

## Accessibility checks

Browser-level semantic checks passed on both routes:

- exactly one `h1` per page;
- no images missing `alt` text;
- no unnamed buttons or links;
- no duplicate element IDs;
- mobile navigation remained available at 390 px;
- export controls were enabled and had visible names;
- focus-visible styles and native button/select semantics were retained.

The final demo-card image pass marks the first visible image as priority-loaded. The earlier development-only Next.js LCP warning was therefore removed without changing content or layout.

## Export review

The benchmark page exposes enabled JSON, CSV, and PDF controls. Serializer tests verify that all three formats are generated from the same normalized records, preserve provenance/status fields, and do not substitute fabricated values for missing metrics.

## Scientific presentation review

- Status badges distinguish verified test, verified validation, smoke-only, operational-only, and unavailable evidence.
- Validation results are not labeled as test results.
- Operational latency/parity evidence is not presented as quality accuracy.
- Missing benchmarks render an explicit evidence-gap state.
- The comparison panel blocks incompatible task/split comparisons.
- Demo cases distinguish annotated ground truth from curated expected behavior and retain source attribution.

## Screenshot index

| File | Review target |
|---|---|
| `report_assets/benchmark-overview-1440.png` | Wide desktop overview |
| `report_assets/benchmark-overview-1280.png` | Standard desktop overview |
| `report_assets/benchmark-overview-1024.png` | Compact desktop/tablet layout |
| `report_assets/benchmark-overview-mobile-390.png` | Mobile benchmark overview |
| `report_assets/rsvqa-model-card.png` | Verified-test model card |
| `report_assets/grounding-model-card.png` | Verified-validation model card |
| `report_assets/changerex-model-card.png` | Operational-only model card |
| `report_assets/sar-translation-card.png` | SAR operational/missing-quality presentation |
| `report_assets/missing-benchmark-state.png` | Explicit unavailable-evidence state |
| `report_assets/benchmark-comparison.png` | Compatibility guard |
| `report_assets/performance-comparison.png` | Runtime comparison |
| `report_assets/export-panel.png` | JSON/CSV/PDF controls |
| `report_assets/demo-gallery-desktop.png` | Desktop demo gallery |
| `report_assets/demo-gallery-mobile.png` | Mobile demo gallery |
| `report_assets/demo-detail.png` | Expanded demo evidence and attribution |

## Result

The module is visually consistent with the existing SatQuery workspace, responsive across the reviewed breakpoints, semantically accessible in the audited surface, and explicit about the strength and limits of every displayed result.
