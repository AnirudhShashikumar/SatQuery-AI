# Single SAR Pix2Pix automatic-routing smoke

- Result: PASS
- Sample: `/Users/anirudhshashikumar/Documents/Projects/SatQuery AI/SAR images/sar 2.png`
- Original query: `Is there water?`
- Detection: `sar_preview` (medium)
- Detection reason: One-channel display has SAR-like intensity statistics and a SAR metadata/filename hint; classified as SAR preview (intensity range p1-p99=244.0, coefficient of variation=0.299, local roughness=0.045).
- Translator: `Pix2Pix`
- Optical specialists: `rsvqa_vqa_specialist, satquery_vision_encoder_v1`
- Response status: `partial` / `COMPLETED_WITH_LIMITATIONS`
- Shared model load count: 0 → 0
- Shared model reuse count: 0 → 1

## Checks

- PASS — upload_accepted
- PASS — automatically_detected_as_sar
- PASS — no_manual_modality_override
- PASS — pix2pix_selected
- PASS — generated_preview_created
- PASS — original_query_processed
- PASS — normal_response_returned
- PASS — required_trace_present
- PASS — no_failure
- PASS — no_duplicate_model_load
- PASS — shared_model_reused

The generated preview is an optical-like Pix2Pix representation, not observed optical truth.
