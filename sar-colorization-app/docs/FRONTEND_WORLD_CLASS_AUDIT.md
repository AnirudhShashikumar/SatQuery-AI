# SatQuery AI Frontend World-Class Audit

## Scope and guardrails

This audit covers the existing Next.js frontend under `frontend/`, with particular attention to the Assistant workspace, its upload and modality-inspection path, the evidence-first result experience, navigation, health display, exports, accessibility, responsiveness, and tests. The implementation is frontend-only. Existing FastAPI endpoints, request and response schemas, routing, inference, model checkpoints, analytics, health contracts, and public field names will remain unchanged.

## 1. Current user journey

1. A user enters `/assistant` through the product sidebar or landing page.
2. The Assistant shows an atmospheric hero, a short introductory message, four intent presets, optional approved demo samples, and a message composer.
3. Upload controls are initially hidden behind “Attach imagery.” The first upload is inspected by `/api/agent/inspect`; a second upload changes the mode to a paired workflow.
4. The workflow can be changed in a disclosure named “Override detected workflow.” Single-image modality can also be confirmed or overridden after inspection.
5. A query is submitted to the existing agent image endpoint. The frontend advances through an estimated stage list while the request runs.
6. The response is rendered by `AssistantResultExperience`, with a summary, evidence viewer, metrics, rationale, limitations, execution timeline, report actions, and an expandable scientific record.
7. Task-specific panels render grounding candidates, VQA evidence, SAR evidence, cross-modal evidence, or bi-temporal evidence using only response data.

The journey is functionally rich, but the primary workflow choice and input requirements are not visible soon enough. The page initially reads more like a chat assistant than a geospatial analysis workspace.

## 2. Current component structure

- `app/[workspace]/page.tsx` maps workspace routes to `Workspace`.
- `Workspace` selects Assistant, model workspaces, reports, settings, and research views.
- `ProductShell` combines a persistent responsive `Sidebar` with page content.
- `AssistantWorkspace` owns local state, inspection, mode selection, file/date inputs, query submission, progress, fallbacks, result routing, and demo loading.
- `AssistantUploadCard` owns dropzone interaction and input metadata display.
- `AssistantResultExperience` owns the executive result hierarchy, evidence viewer, metrics, limitations, timeline, reporting, research view, and technical record.
- Task-specific result panels remain in `AssistantWorkspace`.
- API calls are centralized in `services/api.ts`; domain contracts are centralized in `types/agent.ts`.

The existing state-management approach is appropriate. No global store is needed. The largest maintainability issue is that `AssistantWorkspace` mixes orchestration with several presentational systems, while `AssistantResultExperience` is a second large component with task-specific data derivation.

## 3. Current design system

The app uses Tailwind utilities plus a large `globals.css` file. It already defines dark/light semantic foundations, shared panels, a restrained graphite/cyan palette, responsive sidebar behavior, focus rings, and result-specific styling. Lucide is the single icon system and Next Themes controls appearance.

Strengths:

- restrained dark graphite surfaces and quiet borders;
- clear cyan/green/amber/rose status language;
- compact type and high information density;
- consistent rounded geometry;
- useful light-theme overrides;
- no dependency on a heavy component framework.

Gaps:

- semantic tokens do not yet cover optical, SAR, translated evidence, grounding, change analysis, hover, and skeleton states;
- some CSS still uses raw zinc/sky values rather than semantic tokens;
- typographic roles exist informally rather than as a declared scale;
- “glass,” “panel,” and feature-local cards overlap in responsibility.

## 4. Visual inconsistencies

- The hero and assistant-message treatment suggests a chatbot, while the result is a dense scientific workspace.
- Workflow modes are hidden in a disclosure despite being the most consequential input decision.
- Preset buttons represent intents, while the sidebar and hidden override represent modes; this creates two competing navigation models.
- Upload metadata is thorough but visually dense and not prioritized into identity, compatibility, and advanced facts.
- Result sections have stronger hierarchy and compositional discipline than the input state.
- Health appears in both the sidebar and hero as a simple binary state, but neither provides compact specialist detail.
- Status labels vary between backend-native values, title-cased values, and bespoke copy.

## 5. UX problems

- Users cannot immediately see the three supported workflows and their input requirements.
- The first actionable upload control requires opening an attachment panel.
- Pair roles become clear only after mode selection; examples and dates are separated from the mode affordance.
- The current progress display is useful but has no elapsed-time readout and can imply deterministic stage completion while the backend provides one blocking request.
- Errors are usually a single backend-derived sentence and do not consistently explain what remains available or how to recover.
- The result summary is strong, but evidence-source participation is distributed across products, provenance, and the technical record instead of being scannable in one place.
- Health is technically present but not useful enough for a live evaluator to understand ready, unloaded, disabled, loading, failed, or unavailable states.

## 6. Accessibility problems

- Navigation has a good mobile focus trap and Escape handling, and global focus-visible styles exist.
- The dropzone uses a button and has good labels, but it does not expose a concise compatibility/status summary outside visual metadata.
- Mode controls inside a disclosure are keyboard accessible, yet their hidden placement weakens discoverability.
- Viewer controls are labeled, but image comparison semantics and generated-image disclosures must remain adjacent to the affected visual.
- Some status meaning relies heavily on color, especially small dots and border tones.
- The stage list needs explicit status text for screen readers and an elapsed-time announcement that does not update excessively.
- Motion-reduction coverage is incomplete for pulsing and transition-heavy elements.

## 7. Responsive-layout problems

- The shell and sidebar respond well from desktop to mobile.
- The input composer is optimized around a single column and becomes long when paired uploads, inspection, dates, and compatibility are expanded.
- Workflow controls do not establish a stable compact grid at 1024–1440 px.
- Dense result metric grids can become visually repetitive at laptop sizes.
- Task-specific diagnostic tables are usable but require careful overflow containment.
- Mobile can review results, but paired imagery needs explicit stacking and labels rather than relying on incidental grid behavior.

## 8. Data available but not surfaced well

- `AgentHealth` supports per-specialist status, device, error, enabled state, load count, and reuse count, but the primary Assistant does not expose it.
- Image inspection exposes auto-detected modality, confidence, reason, user confirmation, effective modality, representation, dimensions, bands, georeferencing, and warnings. The information exists but is split between upload metadata and a later inspection block.
- Execution steps expose tool, status, duration, and permitted parameters. The current timeline surfaces only the compact label and runtime.
- Grounding responses contain accepted and rejected candidates, quality fields, source scores, score mode, and limitations; these should read as a detection review, not a generic result table.
- SAR translation includes generated and normalized previews plus disclosures and limitations. The disclosure must remain inseparable from generated imagery.
- Cross-modal statistics, previews, regions, method details, and warnings are available but need a concise agreement state.
- Bi-temporal outputs include engine mode, learned and deterministic masks, comparison metrics, regions, alignment, fallbacks, and warnings; the engine actually used should be more prominent.

## 9. Components worth preserving

- `Sidebar` and its responsive/focus behavior.
- `AssistantUploadCard` dropzone and backend-derived metadata handling.
- `AssistantResultExperience` evidence-product derivation, evidence viewer, metrics, report actions, research view, and disclosure-first behavior.
- `MissionReportActions` and existing report API integration.
- Task-specific task panels and response contracts.
- `services/api.ts`, current hooks/state approach, recent-conversation handling, approved demo workflow support, and all current backend-driven fallbacks.

## 10. Components requiring refactoring

- `AssistantWorkspace`: separate the visible workflow/input shell from request orchestration and make mode selection first-class.
- `AssistantExecutionProgress`: add observable elapsed time, semantic statuses, and fallback/warning language without fake percentages.
- `AssistantUploadCard`: add role/status hierarchy, replace action, compact metadata priority, and compatibility semantics.
- `AssistantResultExperience`: add a concise evidence-source participation panel and improve technical-detail discoverability while preserving current result ordering.
- `Sidebar`: preserve navigation, but make compact health expandable instead of only binary.
- task-specific result panels: tighten headings and ensure generated/observed evidence and accepted/rejected grounding candidates are visibly distinct.

## 11. New components required

The smallest useful reusable set is:

- `ModeSelector`: visible segmented workflow cards with requirements and examples.
- `WorkspaceHeader`: restrained product context, active workflow, and compact health.
- `SystemHealthPanel`: expandable per-specialist states using the existing health endpoint.
- `EvidenceSources`: derives used, fallback, skipped, or unavailable source rows only from published response and execution data.
- `ErrorState`: human-readable recovery guidance around an existing error message.

Existing upload, result, viewer, and report components should be improved rather than duplicated.

## 12. Frontend-only implementation plan

1. Extend the semantic token set and typographic/layout primitives.
2. Replace the chat-led Assistant opening with a geospatial workspace header and visible workflow selector.
3. Make the input workspace visible and role-driven, while preserving existing file inspection, demo loading, cache behavior, and API calls.
4. Improve the query composer with mode context, keyboard guidance, and contextual suggestions.
5. Enhance stage progress with elapsed time and explicit semantic state text.
6. Add compact expandable specialist health using the existing agent-health endpoint.
7. Add evidence-source participation to the result hierarchy and tighten task-specific visual semantics.
8. Add human-readable error recovery and reduced-motion/responsive behavior.
9. Add focused behavior tests, then run the complete frontend suite, lint, and production build.
10. Inspect rendered desktop, laptop, tablet, and mobile states and store review assets under `artifacts/frontend_world_class_review/`.

## 13. Contract confirmation

This redesign will not change backend behavior, endpoints, request schemas, response schemas, Pydantic models, router behavior, inference logic, model behavior, report generation, analytics, health contracts, or public response field names. All new presentation will be derived from fields already published by the production API. No confidence, metric, engine status, model use, or evidence source will be fabricated.
