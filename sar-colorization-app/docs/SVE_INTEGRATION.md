# SatQuery Vision Encoder Integration

## Lifecycle and device behavior

`SVE_ENABLED=true` enables the optional service. Loading is lazy unless `SVE_LOAD_ON_STARTUP=true`. A process-wide, lock-protected service transitions through `unloaded`, `verifying`, `loading`, `ready`, `failed`, or `disabled`; a failed instance is retried only through the explicit service retry operation.

Automatic device order is CUDA, Apple MPS, then CPU. Inference uses float32 on every device. An MPS unsupported-operation failure moves the persistent instance to CPU, retries once, returns an explicit fallback notice, and increments both MPS- and CPU-fallback counters. CPU is supported with bounded batches and caches. `SVE_TIMEOUT_SECONDS` is an advisory runtime bound recorded in the result; PyTorch inference is not force-killed because terminating an active kernel is unsafe.

Configuration:

```env
SVE_ENABLED=true
SVE_DEVICE=auto
SVE_MODEL_DIR=models/satquery_vision_encoder_v1
SVE_LOAD_ON_STARTUP=false
SVE_MAX_BATCH_SIZE=8
SVE_CACHE_SIZE=128
SVE_TIMEOUT_SECONDS=30
```

## Workflow integration

- Captioning: the existing RSICD BLIP captioner remains authoritative. One candidate receives a consistency score; multiple candidates are reranked without rewriting them. Original order, reranked order, selected index, and scores are retained.
- Controlled VQA: the deterministic answer remains unchanged. SVE adds `agreement`, `weak agreement`, `disagreement`, or `unavailable` evidence against controlled scene concepts.
- Grounding DINO: model internals and accepted/rejected boxes are unchanged. SVE may add a weak environment-plausibility warning and never vetoes a detection.
- Routing: SVE records advisory scene evidence with `influence_applied=false`; explicit user intent and deterministic specialist selection remain unchanged.
- Optical/SAR: raw SAR is not encoded. The optical image may receive scene priors. SARFusionFormer output is labelled generated RGB-like imagery; when reference optical is supplied, the API adds “Optical-to-generated-RGB semantic consistency” as supporting evidence only.
- Bi-temporal: eligible before/after optical images receive cosine similarity and top controlled-prior differences. TTP, deterministic masks, and spatial products are unchanged.

All new response fields are optional. Standard responses and reports do not contain raw embeddings, local paths, secrets, checkpoint paths, stack traces, or hidden reasoning.

## Cache and security

Bounded LRU caches store CPU embedding tensors only. Image keys contain a content hash, adapter checksum, backbone ID, and preprocessing-metadata fingerprint. Text keys use normalized bounded text plus the same model identity. Raw images, filenames, filesystem paths, secrets, and full queries are not cached. Analytics retain only counters, device labels, runtimes, fallback counts, and cache rates.

## Local commands

```bash
cd /path/to/sar-colorization-app
venv/bin/python -m pip install -r requirements-backend.txt
env SVE_ENABLED=true SVE_DEVICE=auto venv/bin/python scripts/validate_sve_runtime.py --output artifacts/sve_validation.json
env SVE_ENABLED=true SVE_DEVICE=auto venv/bin/python -m uvicorn backend:app --host 127.0.0.1 --port 8010
cd frontend && npm run dev
```

The first execution may download the generic OpenCLIP pretrained backbone. The local adapter is still verified independently before application.
