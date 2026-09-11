# SIH Final Submission Audit

Run the read-only automated audit from the repository root:

```bash
python3 scripts/final_submission_audit.py --run-tests --output artifacts/final_submission_audit.json
```

## Required gates

- [x] Full backend pytest passes: 573 passed, 5 environment/checkpoint-dependent skips.
- [x] Full frontend unit/integration suite passes: 193 tests across 33 files.
- [x] TypeScript `--noEmit` passes.
- [x] ESLint passes with 0 errors; 27 `no-img-element` warnings are recorded, not called clean.
- [x] Next.js production build passes and generates all 13 routes.
- [x] API/Pydantic schema and canonical benchmark JSON validation pass (9 records plus demo gallery).
- [x] Required checkpoint files and known SHA-256 values pass.
- [x] No private absolute developer paths occur in distributable code/docs/config.
- [x] No apparent API keys or secrets occur in tracked text.
- [x] No accidental large files occur outside declared model/artifact/dependency locations.
- [x] Demo manifest paths, attributions and licenses validate.
- [x] A fresh production-path smoke result exports an authoritative JSON report; broader report-format contracts remain covered by the backend suite.
- [x] Model health exposes explicit lifecycle states.
- [x] README command coverage is checked by the audit; clean-install execution remains a release-machine action.
- [x] Provenance inventory uses “license verification required” rather than guessing unknown licenses.
- [x] All five Problem Statement 26167 representative queries route to the intended workflow.
- [x] Offline production-path smoke passes for single-image, Optical+SAR and bi-temporal workflows.

Automated audit result on 2026-09-03: **PASSED**. The machine-readable record is `artifacts/final_readiness/audit.json`; the workflow smoke is `artifacts/final_readiness/workflow_smoke.json`. Browser-based visual inspection was attempted after the production build, but the browser-control security policy blocked localhost navigation. UI acceptance therefore rests on 193 component/integration tests, TypeScript, and the successful production build; do not describe this run as a completed manual visual review.

## Evidence review

Verified RSVQA and grounding records must retain their original dataset, split, sample count and metrics. ChangerEx parity, single-image translation smoke, latency runs and manually selected demos must not appear as accuracy. Missing CDVQA, LEVIR-CD, paired SAR translation, caption, or controlled Optical+SAR runs remain `unavailable`/`operational_only` until their labelled protocols complete.

## Human actions before packaging

1. Confirm every upstream model/dataset license from its authoritative release and include required notices.
2. Supply and run approved CDVQA and LEVIR-CD official test splits.
3. Supply a held-out paired SAR/optical dataset and controlled Optical+SAR references.
4. Run representative Cartosat-2S/RISAT fixtures when access is authorized.
5. Review the generated audit artifact and all warnings; do not submit with unresolved secrets, paths or hash mismatches.
