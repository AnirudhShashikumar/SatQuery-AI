# SatQuery Architecture

## Purpose

SatQuery is GeoVision’s local orchestration and evidence layer. It validates imagery, selects a supported specialist deterministically, executes the specialist, and returns typed results with provenance, scientific warnings, confidence rationale, and an observable trace.

## Request flow

```text
Next.js Assistant
  → bounded multipart ingestion
  → file-signature and modality validation
  → Rasterio/tifffile/Pillow metadata and safe preview
  → deterministic query routing
  → compatible specialist
      ├─ RSICD-adapted optical captioner
      ├─ local zero-shot Grounding DINO optical box grounder
      ├─ deterministic optical evidence + controlled VQA
      ├─ deterministic bi-temporal change engine + controlled Q&A
      └─ deterministic optical–SAR evidence fusion + controlled Q&A
  → typed result, evidence, confidence, provenance, trace
  → bounded request-ID result store
  → backend-authoritative PDF/JSON/CSV/ZIP report
```

Pix2Pix and SARFusionFormer remain independent reconstruction endpoints and are not silently invoked by SatQuery routing.

## Input and evidence handling

- PNG, JPEG, TIFF, and GeoTIFF are validated from file bytes, not extensions alone.
- Each accepted upload receives a SHA-256 digest used only for safe cache identity.
- Original upload bytes are not retained by the mission result cache.
- Source and evidence previews are UUID-named PNGs served through a restricted route.
- Pair workflows never silently register, reproject, or resample imagery.

## Result and analysis cache

The process-local store is bounded by item count and expiry. The cache key covers both input hashes, routed task, normalized query, modalities, dates/safe parameters, and tool version. Cache reuse is opt-in, clearly labelled, and bypassed by `force_rerun`.

Restarting the API clears result/cache records. Expiry or eviction removes their temporary preview products. No database is used.

## Mission reports

`POST /api/agent/report` accepts a stored request ID and desired formats. It does not accept arbitrary frontend statistics. Generated artifacts are temporary and served as attachments:

- PDF: white-page mission document with embedded evidence and paginated trace.
- JSON: complete versioned, typed mission document and authoritative response.
- CSV: numeric statistics and connected-region rows.
- ZIP: PDF, JSON, CSV, evidence PNGs, and README.

## Remote-sensing adaptation

The system’s trained remote-sensing adaptation is the separate RSICD-fine-tuned BLIP captioner. Grounding DINO is an official local zero-shot open-vocabulary detector and is truthfully marked `remote_sensing_adapted: false`; its boxes and alignment scores are model-produced evidence, not ground truth. Controlled VQA is a deterministic evidence specialist and is also marked `remote_sensing_adapted: false`. No RSVQA or remote-sensing Grounding DINO fine-tuning is claimed.

## Security and privacy boundary

Reports exclude original uploads, API keys, environment values, private cache locations, absolute local paths, and hidden reasoning. Provider settings and API-key handling are not part of SatQuery reporting or demo mode.
