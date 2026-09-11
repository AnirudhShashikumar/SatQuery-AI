# SatQuery product elevation audit

Audit recorded before implementation, 5 September 2026. Scope is the existing application; model inference, checkpoint files and benchmark records remain protected.

## Architecture and evidence flow

The Next.js workspace sends inspected observation files to FastAPI. Ingestion preserves representation and source identity; routing selects existing specialists. Compatibility gates quantitative pair analysis. The cross-modal analyzer independently prepares optical and native SAR evidence, permits same-sized unregistered preview pairs only for qualitative reasoning, and reserves joint statistics for exact grids. VQA composes deterministic answers from published evidence. Change analysis combines the existing learned detector with independent deterministic evidence and evidence-gated semantic interpretation. Reports and comparisons preserve source identity. Existing model verification, compliance suites, workflow smoke and final submission audit cover these boundaries. No inference replacement is warranted.

Reviewed the workspace, upload cards, result experience, evidence mapping, navigation, shell, landing, presentation scenes, history persistence, report exports, benchmark overview/cards, compliance matrix, router, compatibility/ingestion flow, VQA composition, cross-modal result generation and final audit tooling. The existing launcher already starts and warms the local backend; no new mandatory service or memory-heavy prewarming is needed.

## P0 — correctness and broken UX

- Progress advances every 850 ms and labels guessed specialists active. Those transitions are not backend observations. Replace with request-in-flight status and elapsed time; completed trace remains authoritative.
- Upload cards label every inspected file compatible/verified even if content is ambiguous or placed in the wrong role. Separate file inspection from role validation.
- Judge confidence can render raw scores as percentages without explaining calibration. Lead with qualitative level and the published reason.
- Report library is wired only to reconstruction-session state despite normal assistant report downloads working. Add the existing stored-analysis catalogue as the primary library.
- Existing change summary derives compass directions from pixel coordinates. Use image-relative terminology unless orientation is independently established.
- Light-theme viewer source label is nearly invisible on its dark image canvas. Isolate source-label contrast from global light text overrides.

## P1 — analyst usefulness

- Judge View shows the full SVE panel, evidence-source catalogue, metrics, rationale and execution timeline. Move these behind Research View; preserve complete record and exports.
- Cross-modal answers list almost every sector and lead with a workflow announcement. Compose query-specific statements, compress complete rows/columns and widespread sector support without implying mask coverage.
- No compact structured findings connect optical evidence, native SAR evidence and their joint interpretation. Add findings derived only from existing facts, source links and published region geometry.
- Viewer offers Layer/Opacity/Split, has large unused canvas space and no simultaneous source-specific comparison. Add visibly labelled side-by-side and clarify blend semantics.
- History stores only five prompt strings and does not reopen authoritative results. Use the existing backend catalogue for completed analyses, preserving prompt history as a fallback.

## P2 — professional polish

- The first viewport repeats workflow context four times. At 1366×768 the observation and query panes begin below the fold; even 1536×864 does not show Run Analysis.
- Large hero, nested outlined cards, repeated uppercase labels, six suggestion chips and repeated ready badges compete with imagery.
- Sidebar mixes workspace/research destinations and exposes three advanced disclosure groups. Consolidate into Analyze, Workspace, Research and one collapsed Advanced group; remove Research Cloud branding.
- Landing uses animated orbit/glow decoration and an unverified Agent online label. Replace with a calm workflow launch surface.
- Compliance table requires a 1280 px interior width and seven dense columns. Use five meaningful columns with expandable supporting detail.
- Presentation already runs real workflows and has a dedicated shell. Preserve that mechanism, add a concise real-workflow presentation entry and keep provenance and limitations accessible.

## P3 — optional, deliberately bounded

No new VLM, semantic mask, database platform, screenshot framework, speculative inference optimization or launcher memory increase. Reuse current components and measured benchmark data. Retain light mode and reduced-motion support.

## Browser inspection before implementation

Opened the actual running Optical + SAR workspace and completed-result state. Inspected light and dark screenshots at 1366×768, 1440×900, 1536×864 and 1920×1080. Reviewed upload roles, answer length, source viewer, Judge/Research controls and expanded trace through DOM and screenshots. The source-specific image mapping from the previous repair remains correct. Remaining routes and state combinations will be inspected during implementation and final verification; these are not claimed as completed here.

## Implementation and acceptance plan

1. Truthful progress, compact workspace and role status.
2. Query-aware conservative answer composition, answer-first result, structured findings and source comparison.
3. Shared restrained design layer, navigation, landing, authoritative history/reports, benchmark grouping and compliance matrix.
4. Presentation and accessibility review; component and scientific regression tests after each subsystem.
5. Actual browser workflows, requested dimensions, full pytest/frontend/type/lint/build, PS 26167, model verification and final submission audit. Record evidence and remaining limits in ASTRA_PRODUCT_ELEVATION_REPORT.md.

## Final scope amendment — 6 September 2026

The user subsequently required the original UI to remain unchanged and asked for the remaining work to continue. That instruction supersedes the frontend implementation plan above. All UI changes from the elevation pass were removed, including the new stylesheet, revised navigation, landing page, presentation controls, viewer layout and history library. The prior Optical + SAR repair was preserved. The complete frontend source digest exactly matches the recorded pre-elevation digest: `3106b80e5e6cb669b2a48e31c6fa4e3702aaf62cca3231c5d4fe7fdca20b3e09`.

The audit findings above remain historical findings, not claims of completed frontend repairs. Backend answer composition, metadata routing, image-relative change descriptions, stored-result retrieval, tests and documentation comprise the final elevation scope. See [the completion report](ASTRA_PRODUCT_ELEVATION_REPORT.md) for results and explicit remaining limitations.
