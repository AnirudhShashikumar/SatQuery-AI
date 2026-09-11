# SatQuery Offline Demo Runbook

## Preflight

1. Confirm Pix2Pix and SARFusionFormer checkpoints are at their configured local paths.
2. Cache the RSICD caption checkpoint before disconnecting from the internet if captioning will be demonstrated.
3. Verify the five files under `satquery_agent/demo_samples/` exist.
4. Start the backend with demo mode enabled:

```bash
SATQUERY_DEMO_MODE=true venv/bin/uvicorn backend:app --host 127.0.0.1 --port 8010
```

5. Start the frontend:

```bash
cd frontend
npm run dev
```

6. Open `http://127.0.0.1:3000/assistant` and verify the “Approved offline samples” panel appears.

## Exact judge sequence

### 1. Single-image understanding

1. Click `Load Demo Sample · Single-image understanding`.
2. Keep the prefilled water question and click Analyze.
3. Show category, evidence percentages, five evidence products, regions, limitations, confidence, and trace.
4. Download PDF, JSON, or Full ZIP.
5. Optionally switch the question to scene description. This requires the cached caption checkpoint.

### 2. Bi-temporal change

1. Load the bi-temporal sample; files and valid dates are populated locally.
2. Click Compute Change.
3. Show before, after, difference, mask, overlay, changed percentage, region count, controlled answer, and trace.
4. Download Full ZIP and point out the embedded products and CSV.

### 3. Optical–SAR analysis

1. Load the optical–SAR sample.
2. Click Analyze.
3. Show exact compatibility, modality-specific evidence, joint evidence, agreement/disagreement statistics, regions, confidence, and trace.
4. Download the report.

### Technical Q&A

Open `/assistant/compliance`. Show the official requirement matrix and tool registry. Highlight that Grounding DINO is a local zero-shot optical grounder rather than a remote-sensing-fine-tuned model, controlled VQA is deterministic, and the RSICD captioner is the trained remote-sensing-adapted component.

## Offline checklist

- Backend Python environment and frontend dependencies are already installed.
- All three reconstruction checkpoints are local if their separate pages are demonstrated.
- RSICD caption checkpoint is cached if captioning is demonstrated.
- Grounding DINO tiny is cached and `SATQUERY_GROUNDER_LOCAL_FILES_ONLY=true` is set if grounding is demonstrated offline.
- `SATQUERY_DEMO_MODE=true` is set only for the local demo process.
- No Gemini or other hosted provider is required for SatQuery workflows.
- Browser and backend use `127.0.0.1`; no sample is fetched from the internet.

## Expected timings

On the checked-in 128-pixel samples, deterministic VQA/change/fusion usually completes in well under one second on a normal laptop; actual times are shown in every trace. Report generation is normally a few seconds or less. Captioning and Grounding DINO first load are device- and cache-dependent and can take substantially longer; subsequent reuse is disclosed in the result.

These are demo expectations, not benchmark results.

## Fallback procedure

- If the backend restarted and a report says its result expired, rerun the loaded sample and generate the report again.
- If the caption checkpoint is absent offline, demonstrate deterministic VQA instead and show the captioner as unavailable in the registry.
- If a pair reports `alignment_required`, use the approved exactly aligned demo sample; GeoVision will not silently align user imagery.
- If cached output is shown, use `Re-run Analysis` to bypass reuse and demonstrate a fresh execution.
