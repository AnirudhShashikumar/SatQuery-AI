# SatQuery SIH 26167 Compliance

The live authoritative matrix is available at `/assistant/compliance` and `GET /api/agent/compliance`.

| Official requirement | Implementation | Status | Evidence | Limitation |
|---|---|---|---|---|
| Single optical image | Validated optical/multispectral ingestion | Available | Ingestion, caption, VQA tests | RGB-like interpretation required for semantic workflows |
| Single SAR input | SAR metadata and safe preview ingestion | Available | Ingestion modality tests | SAR captioning and single-image SAR VQA are unavailable |
| TIFF/GeoTIFF | tifffile plus Rasterio metadata | Available | TIFF/GeoTIFF tests | Preview is display-only |
| Captioning | RSICD-fine-tuned BLIP | Available when checkpoint configured | Caption provenance and optional checkpoint test | Offline first use requires cached checkpoint |
| Single-image VQA | Deterministic evidence-grounded templates | Available | Controlled VQA tests | Not RSVQA-fine-tuned or calibrated |
| Bi-temporal analysis | Normalized difference, morphology, regions | Available | Change tests | No automatic registration or causal semantics |
| Change VQA | Controlled templates over real change statistics | Available | Change-VQA tests | No flood/construction/deforestation inference |
| Optical–SAR analysis | Deterministic aligned evidence fusion | Available | Cross-modal tests | Not calibrated backscatter or semantic truth |
| Agent routing | Deterministic task rules | Available | Router tests | Unsupported queries fail explicitly |
| Registry | Public typed tool registry | Available | Registry tests | Unavailable tools remain visible |
| Evidence | UUID previews, masks, overlays, regions | Available | Preview tests | Model/heuristic evidence, not ground truth |
| Confidence | Rationale and real scores only | Available | Contract tests | Heuristic semantic confidence is uncalibrated |
| Execution trace | Ordered stages, safe parameters, timings | Available | Trace tests | Timings are device-dependent |
| Mission report | Backend PDF/JSON/CSV/ZIP | Available | Readiness artifact tests | Bounded temporary lifetime |
| Remote-sensing adaptation | RSICD captioner | Available when checkpoint configured | Caption provenance | VQA itself is deterministic |
| Text-guided grounding | Official local Grounding DINO tiny zero-shot detector | Optional / Available | Unit, API, report, and real checkpoint tests | Optical/RGB-like multispectral only; boxes are not ground truth; no mask refinement |

## Mandatory status

The implemented MVP covers the judge-facing workflows and exposes real evidence, provenance, confidence, traces, and reports. Optional grounding runs the official checkpoint locally and reports model-produced boxes and alignment scores without claiming remote-sensing fine-tuning. GeoVision does not claim complete SAR VQA, public benchmark performance, calibrated segmentation, or operational ISRO deployment.
