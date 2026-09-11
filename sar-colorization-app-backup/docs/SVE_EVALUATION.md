# SatQuery Vision Encoder v1 Evaluation

The table below reproduces the verified validation comparison exactly.

| Metric | Generic OpenCLIP | SatQuery Vision Encoder v1 |
|---|---:|---:|
| loss | 4.8016743659973145 | 4.120570659637451 |
| image-to-text R@1 | 0.0021824531722813845 | 0.005674378015100956 |
| text-to-image R@1 | 0.0008729812107048929 | 0.006110868416726589 |
| image-to-text R@5 | 0.010039283894002438 | 0.022697512060403824 |
| text-to-image R@5 | 0.004364906344562769 | 0.03491925075650215 |
| image-to-text R@10 | 0.021388040855526924 | 0.050632912665605545 |
| text-to-image R@10 | 0.00654735928401351 | 0.05805325135588646 |

Every recorded metric improves relative to the generic baseline on this validation split. The absolute recall values remain low and must not be represented as broad retrieval accuracy.

## Runtime validation sequence

Run:

```bash
env SVE_ENABLED=true SVE_DEVICE=auto venv/bin/python scripts/validate_sve_runtime.py --output artifacts/sve_validation.json
```

The script verifies the checksum, loads the exact adapter, evaluates an approved optical PNG and optical GeoTIFF, runs multi-candidate caption reranking, controlled VQA consistency, grounding-environment support, before/after semantic comparison, raw-SAR skip behavior, model reuse, cache behavior, actual device, runtime, and peak process memory. It emits no raw embeddings.

For the Presentation Mode demo: load the approved single optical sample, show the “Remote-sensing adapted” badge and top priors, run captioning, show caption consistency, ask the land-cover VQA question, show its evidence-consistency state, open the execution trace, and disclose the model provenance and limitations.
