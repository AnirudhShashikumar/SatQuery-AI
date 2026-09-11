# Standalone ChangerEx for macOS

This package runs the official LEVIR-CD ChangerEx ResNet-18 checkpoint using
only PyTorch, NumPy, and Pillow. It is isolated from the SatQuery production
agent and does not import Open-CD, MMCV, MMSegmentation, MMDetection,
MMPretrain, FastAPI, or any SatQuery module.

## Provenance

- Model: ChangerEx with IA-ResNetV1c-18
- Official config: `configs/changer/changer_ex_r18_512x512_40k_levircd.py`
- Open-CD: v1.1.0, commit `09c03eb1077f06191c5448ea1fcf2f2f88ec98c2`
- Checkpoint: `ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth`
- Required SHA-256: `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618`
- Classes: `unchanged` (0), `changed` (1)

Architecture math and module layout were adapted from the Apache-2.0 licensed
Open-CD project, particularly:

- `opencd/models/backbones/interaction_resnet.py`
- `opencd/models/utils/interaction_layer.py`
- `opencd/models/decode_heads/changer.py`
- `opencd/models/necks/feature_fusion.py`
- MMSegmentation's ResNet implementation used by the pinned Open-CD release

Only the selected inference graph is reproduced. The original projects and
their license/copyright notices remain authoritative.

## Install in a new isolated environment

Do not install into the SatQuery production environment.

```bash
python3.11 -m venv .venv-changerex
source .venv-changerex/bin/activate
python -m pip install -r changerex_local/requirements-macos.txt
```

## CLI

```bash
python -m changerex_local \
  --earlier /path/to/before.png \
  --later /path/to/after.png \
  --checkpoint /path/to/ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth \
  --output-dir /path/to/output \
  --device auto \
  --threshold 0.5 \
  --warmup-runs 1 \
  --benchmark-runs 5
```

`auto` verifies MPS with a real tensor operation and selects it when usable;
otherwise it records a warning and selects CPU. An explicit `--device mps`
fails if MPS cannot be verified. CPU fallback for an explicit MPS request is
only permitted with `--allow-device-fallback`.

The CLI writes the probability array and visualization artifacts plus a strict
checkpoint report, environment report, regions, JSON result, and benchmark
report. An error is emitted as bounded JSON on stderr and returns a non-zero
status.

## Python API

Configure the immutable, process-wide lifecycle once, then predict repeatedly:

```python
from changerex_local import configure_lifecycle, predict_change

configure_lifecycle("/path/to/checkpoint.pth", device="auto")
result = predict_change("before.png", "after.png", device="auto", threshold=0.5)
print(result.changed_percentage, result.selected_device)
```

The checkpoint is hashed and loaded on CPU before the model moves to the
selected device. Loading is lazy, thread-safe, strict, and sticky on failure.
Checkpoint and device configuration cannot change after the singleton is
created.

## Preprocessing and outputs

Pillow decodes both inputs directly to RGB. Channels are concatenated as
earlier RGB followed by later RGB, kept in float32 `[0, 255]`, resized with
aspect ratio preserved so the longest side is 1024 by default, normalized by
ImageNet mean `[123.675, 116.28, 103.53]` and standard deviation
`[58.395, 57.12, 57.375]` for each time point, then padded on the right/bottom
with normalized-space zeros to a multiple of 32. This reproduces the official
test resize/data-preprocessor path without its training augmentations.

The two-class logits are bilinearly restored to the processed image size.
Class-1 softmax is the change probability. Probability is bilinearly restored
to source dimensions; the thresholded binary mask is restored with nearest
neighbor. Connected components are 8-connected and use exclusive `(x2, y2)`
bounds. No morphology is applied.

The images must have identical dimensions and be geometrically co-registered.
Outputs are predictions, not ground truth. LEVIR-CD is building-change focused,
so performance on fire, flooding, vegetation, or other domains is not implied
by a successful operational smoke test.
