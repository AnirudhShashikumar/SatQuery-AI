# Final presentation runbook

## Before the event

1. Deploy the service on a persistent CUDA host; never rely exclusively on a temporary Colab tunnel.
2. Run asset verification and start one Uvicorn worker.
3. Confirm `/health` reports `ready`, `checkpoint_verified: true`, device `cuda`, and checkpoint fingerprint `60294429b3d`.
4. Run one warm-up prediction and confirm the next request reports model reuse.
5. Run the benchmark protocol and approve a pair only after reviewing the mask, source evidence, runtime, GPU memory, and fallback.
6. Set GeoVision server-side `TTP_ENABLED=true`, `TTP_DEFAULT_MODE=hybrid`, and the private service connection settings.
7. Preload the approved local pair, verify hybrid analysis and PDF/JSON/CSV/ZIP generation, cache the approved result where permitted, and prepare a recorded backup.

## Exact Stage 8 sequence

1. Open Presentation Mode and advance to Stage 8.
2. Select **Load Approved Pair**. Explain that loading does not start analysis.
3. Point to pair compatibility and dates.
4. Select **Run Change Analysis** to invoke hybrid analysis explicitly.
5. Confirm the **Hybrid Change Analysis · TTP primary** badge.
6. Show Before, After, Difference, TTP learned mask, deterministic mask, Agreement, Disagreement, and TTP overlay.
7. Show TTP changed area, deterministic changed area, mask IoU, agreement, largest region, and runtime. State that IoU is not ground-truth accuracy.
8. Show model, architecture, LEVIR-CD training provenance, checkpoint, CUDA device, and warm-model reuse.
9. Show the compact execution trace and fallback state.
10. State: “GeoVision uses TTP as the primary learned change detector for compatible optical pairs. An independent deterministic engine produces a second evidence stream. Agreement and disagreement are shown transparently, and the system does not infer the semantic cause of change.”
11. Continue to Stage 10 and generate the report. Confirm the report disclaimer that TTP output is not ground truth.

If TTP fails, continue the demonstration with the visible deterministic fallback and explain the displayed reason. Do not restart the full GeoVision request merely to hide a fallback.
