# ChangerEx local-feasibility audit

Date: 2026-09-01  
Scope: standalone local inference audit only; no SatQuery production integration

## Executive decision

The official pretrained ChangerEx ResNet-18 weights are locally viable on this Apple Silicon Mac, on both CPU and MPS, when the selected architecture is extracted into ordinary PyTorch. The unmodified Open-CD 1.1.0 runtime is not a production-safe route on this host: `mmcv-lite` cannot import the required Open-CD/MMSeg module graph, while full MMCV requires a locally compiled `_ext` and failed in two controlled environments.

The recommended route is a small, isolated pure-PyTorch inference package that reproduces only the selected ResNetV1c, exchange layers, Changer decode head, preprocessing, and postprocessing. It must strict-load the original checkpoint and be parity-tested before any later production proposal. This audit did not integrate that package into SatQuery.

## 1. Host evidence

| Item | Observed value |
|---|---|
| Hardware | MacBook Air, model `Mac17,3` |
| Chip / architecture | Apple M5, `arm64`, 10 cores (4 performance + 6 efficiency) |
| Unified memory | 16 GB |
| macOS | 26.6.2, build `25G83`, Darwin 25.6.0 |
| Compiler | Apple clang 21.0.0, target `arm64-apple-darwin25.6.0` |
| System Python | CPython 3.9.6 |
| Other locally available Python | CPython 3.11.15 and 3.12.13 from uv-managed runtimes |
| Existing SatQuery PyTorch | 2.8.0, torchvision 0.23.0 |
| PyTorch MPS built | Yes |
| PyTorch MPS available | Yes; verified outside the filesystem sandbox by allocating a tensor on `mps:0` |

The sandboxed capability probe returned `mps.is_available() == False`; the same read-only probe outside that execution isolation returned `True` and successfully allocated `tensor([1.], device='mps:0')`. All reported MPS execution measurements below were run outside that isolation.

Existing environments found before the audit:

- `sar-colorization-app/venv`: CPython 3.9.6, PyTorch 2.8.0; inspected read-only and not modified.
- sibling `../.venv`: CPython 3.11.15; inspected read-only and not modified.
- sibling backup `../sar-colorization-app-backup/venv`: CPython 3.9.6; inspected read-only and not modified.

All Open-CD installation and execution experiments used disposable environments under `/private/tmp/changerex-local-audit-20260831/`.

## 2. Exact model and checkpoint

| Item | Selection |
|---|---|
| Project | [Open-CD](https://github.com/likyoo/open-cd) |
| Open-CD version | `v1.1.0` |
| Pinned commit | [`09c03eb1077f06191c5448ea1fcf2f2f88ec98c2`](https://github.com/likyoo/open-cd/tree/09c03eb1077f06191c5448ea1fcf2f2f88ec98c2) |
| Config | [`configs/changer/changer_ex_r18_512x512_40k_levircd.py`](https://github.com/likyoo/open-cd/blob/v1.1.0/configs/changer/changer_ex_r18_512x512_40k_levircd.py) |
| Architecture | `DIEncoderDecoder` + `IA_ResNetV1c` depth 18 + `Changer` head (`ChangerEx`) |
| Training data | LEVIR-CD |
| Training schedule | 40,000 iterations; 512×512 training crop |
| Classes | index 0 `unchanged`; index 1 `changed` |
| Official checkpoint | `ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth` |
| Official source | [Open-CD Changer model table](https://github.com/likyoo/open-cd/tree/v1.1.0/configs/changer), [Google Drive file `1SZ8DBgBryJ63X9AU0AQvaJ2ZH3oQHQHe`](https://drive.google.com/file/d/1SZ8DBgBryJ63X9AU0AQvaJ2ZH3oQHQHe/view) |
| Checkpoint size | 136,880,383 bytes (about 130.5 MiB) |
| SHA-256 | `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618` |

The checkpoint contains `meta`, `state_dict`, and `optimizer`. Its metadata identifies the original training-era stack as MMCV 1.6.2 and Open-CD/MMseg fork version `0.0.2+da8b536`; the current v1.1.0 config is a migrated OpenMMLab 2.x representation. The state dictionary contains 173 tensor/buffer entries and loads strictly into the audited extraction with zero missing and zero unexpected keys.

The checkpoint metadata palette is reversed relative to the documented inferencer example. For unambiguous rendering, use numeric class IDs as authoritative and explicitly set `[0,0,0]` for `unchanged` and `[255,255,255]` for `changed`, as in the official inference example. This does not change the checkpoint or numeric mask.

## 3. Exact config inheritance tree

```text
configs/changer/changer_ex_r18_512x512_40k_levircd.py
├── configs/_base_/models/changer_r18.py
└── configs/common/standard_512x512_40k_levircd.py
    └── configs/common/standard_256x256_40k_levircd.py
        └── configs/_base_/default_runtime.py
```

The selected override sets exchange interactions after the second, third, and fourth ResNet stages, two output classes, OHEM for training, AdamW, and a 512×512 training crop. Inference uses whole-image mode.

## 4. Architecture traced from the imported modules

```text
paired images
  → concatenate as six channels
  → DualInputSegDataPreProcessor
  → split into two RGB tensors
  → shared IA_ResNetV1c-18
      stage 1: identity interaction
      stage 2: SpatialExchange(p=1/2)
      stage 3: ChannelExchange(p=1/2)
      stage 4: ChannelExchange(p=1/2)
  → four pairs of feature maps
  → per-scale 1×1 projections and bilinear resizing
  → feature concatenation and 1×1 fusion
  → FDAF flow alignment using depthwise convolution + grid_sample
  → concatenated aligned differences
  → MixFFN
  → 1×1 two-class classifier
  → resized logits
  → argmax binary mask
```

The parameter-only model has 11,390,946 trainable/state parameters; checkpoint tensor counts are slightly higher because running-statistic buffers are included.

## 5. Input preprocessing and test pipeline

The v1.1.0 inherited test pipeline is:

1. `MultiImgLoadImageFromFile`, replaced by `MultiImgLoadInferencerLoader` by `OpenCDInferencer`.
2. `MultiImgResize(scale=(1024, 1024), keep_ratio=True)` using bilinear interpolation and the OpenCV backend by default.
3. `MultiImgLoadAnnotations`, removed by the inferencer.
4. `MultiImgPackSegInputs`, which converts each HWC image to CHW and concatenates the pair to six channels.
5. `DualInputSegDataPreProcessor` converts BGR-decoded channels to RGB, casts to float32, normalizes, and pads.

Exact normalization, repeated for both dates:

- mean: `[123.675, 116.28, 103.53]`
- standard deviation: `[58.395, 57.12, 57.375]`
- input value scale: 0–255 float pixels, not 0–1
- final ordering: RGB for date A followed by RGB for date B

Padding is on the right and bottom to a divisor of 32. It is applied after normalization with normalized-space `pad_val=0`; segmentation padding is 255. Padding is removed during result postprocessing.

Important naming nuance: `512x512` identifies the training crop/config family. The inherited official test pipeline fits each pair within 1024×1024 while preserving aspect ratio, then pads to a multiple of 32. A square 512×512 input is therefore upscaled to 1024×1024 by the unmodified inferencer. The convolutional model itself accepts other divisor-compatible sizes, and the audit measured both direct 512 and inherited 1024 square execution.

The two images must be spatially co-registered and have the same post-resize shape. The model performs no geometric registration.

## 6. Output contract

- The Changer decode head emits raw two-channel logits at the first backbone feature scale (one quarter of the input spatial dimensions).
- MMSeg's decode-head prediction path bilinearly resizes logits to the processed image shape; model postprocessing removes padding and resizes them to the original image shape.
- A probability map is not returned by default. It is `softmax(final_logits, dim=class)` when explicitly required.
- The binary mask is `argmax(final_logits, dim=class)`: 0 is unchanged and 1 is changed.
- `OpenCDInferencer` normally returns the numeric mask under `predictions`. With `return_datasample=True`, the `SegDataSample` also contains raw `seg_logits` and `pred_sem_seg`.

## 7. Dependency graph and exact audit stack

```text
OpenCDInferencer
├── opencd 1.1.0 @ 09c03eb…
├── mmengine 0.10.7
├── mmsegmentation 1.2.2
│   └── imports mmcv.ops.point_sample through mmseg.utils
├── mmdetection 3.3.0
│   └── imported unconditionally by opencd/__init__.py
├── mmpretrain 1.2.0
│   └── required by eagerly imported Open-CD ViT/TTP modules
├── mmcv 2.0.1 (full build, compiled _ext)
│   └── required by eager mmseg/Open-CD imports
├── torch + torchvision
├── numpy, scipy, matplotlib, prettytable, Pillow/OpenCV
└── ftfy + regex
    └── needed by the installed MMSeg utility import path but not declared by this minimal model
```

Open-CD 1.1.0 enforces these broad runtime ranges in `opencd/__init__.py`: MMCV `>=2.0.0rc4,<2.2.0`, MMEngine `>=0.6.0,<1.0.0`, MMSeg `>=1.0.0rc6,<1.3.0`, and MMDetection `>=3.0.0rc6,<4.0.0`. Its `requirements/mminstall.txt` is stricter for MMCV: `>=2.0.0rc4,<2.1.0`.

Two controlled full-stack variants were tested:

| Variant | Result |
|---|---|
| Python 3.9.6, torch 2.8.0, torchvision 0.23.0, mmcv-lite 2.0.1 | Packages installed, but `opencd.apis` and `opencd.models` failed with `No module named 'mmcv._ext'`. |
| Same stack with full mmcv 2.0.1 | Full MMCV compiled, but loading `_ext` failed with missing symbol `at::mps::MPSStream::commit(bool)`. This matches an upstream-reported macOS linking failure. |
| Python 3.9.6, torch 2.1.2, torchvision 0.16.2, full mmcv 2.0.1 | Full MMCV failed to compile against the current macOS 26 SDK/libc++ (`std::is_arithmetic` specialization rejected in PyTorch headers). |

MMCV documentation distinguishes full `mmcv`, which compiles ops, from `mmcv-lite`, which omits them. The observed missing-symbol failure is also documented in [MMCV issue #2951](https://github.com/open-mmlab/mmcv/issues/2951).

### Is mmcv-lite sufficient?

No, not for the unmodified Open-CD 1.1.0 inferencer. The selected ChangerEx computation does not need MMCV custom ops, but the framework's eager import graph does. MMSeg imports `mmcv.ops.point_sample`, and Open-CD eagerly imports `LightCDNet`, which imports `mmcv.ops.CrissCrossAttention`.

### Is full MMCV required?

Yes for unmodified Open-CD 1.1.0 on this dependency set. That means a compiled `mmcv._ext` must build and dynamically link against the exact PyTorch/SDK ABI. It did not produce a usable stack in either tested combination.

## 8. Custom-op and device audit

### Selected ChangerEx execution path

| Concern | Finding |
|---|---|
| `mmcv.ops` | No selected ChangerEx forward operation calls `mmcv.ops`. It uses `mmcv.cnn` wrappers in the framework implementation. |
| Deformable convolution | Not instantiated. `IA_ResNetV1c` inherits optional `dcn` arguments, but the selected config leaves `dcn=None` and all stages use ordinary convolutions. |
| Modulated deformable convolution | Not used. |
| Custom C++ extension | Not needed by the model math, but required indirectly to import the unmodified framework. |
| Custom CUDA extension | Not used by the target model. No CUDA is available or needed for the extracted model. |
| `SyncBatchNorm` | Configured in backbone and decode head. Evaluation-mode SyncBatchNorm was directly tested successfully on CPU and MPS. A pure extraction can safely instantiate ordinary `BatchNorm2d` for single-device inference because checkpoint keys/running statistics are compatible. |
| Hard-coded CUDA | None in the selected model forward. `default_runtime.py` contains training-only NCCL/cuDNN settings. The old checkpoint metadata records `device='cuda'` and `gpu_ids=[0]`, but this metadata is not executed for inference. |

### Unrelated eager imports that create risk

- `opencd/models/backbones/lightcdnet.py` imports `mmcv.ops.CrissCrossAttention`.
- MMSeg utility initialization imports `mmcv.ops.point_sample`.
- Open-CD ViT/TTP modules import MMPreTrain.
- Full MMCV contains a large suite of compiled CPU/MPS/CUDA-facing ops, including deformable and modulated deformable convolution, even though ChangerEx does not call them.

## 9. Measured CPU and MPS evidence

A disposable pure-PyTorch reconstruction was made from the exact audited module operations and checkpoint key layout. It uses only PyTorch, strict-loads the original `state_dict`, converts the single-device inference normalization layers to `BatchNorm2d`, and makes exchange masks/grid tensors device-local. No checkpoint value was edited.

All runs used batch size 1, float32, a normalized synthetic paired RGB tensor, one warm-up, and synchronized MPS timings. The result is an execution/compatibility measurement, not an accuracy benchmark.

| Device | Spatial size | Mean warmed forward | Observed memory |
|---|---:|---:|---:|
| CPU | 512×512 | 130.2 ms over 5 runs | 619.5 MB process peak RSS |
| MPS | 512×512 | 39.4 ms over 5 runs | 384.7 MB process peak RSS; MPS allocation not captured in this first run |
| CPU | 1024×1024 | 535.3 ms over 3 runs | 1,437.6 MB process peak RSS |
| MPS | 1024×1024 | 133.2 ms over 3 runs | 535.3 MB process peak RSS; about 1,146 MB MPS driver allocation in a follow-up run |

For every run:

- strict load reported zero missing and zero unexpected keys;
- output logits had shape `[1, 2, H, W]` after final resize;
- probabilities had the same shape and summed to approximately 1 per pixel;
- the argmax mask had shape `[1, H, W]`.

The 512 probe's raw random input produced only class 0. That is not a quality statement; real imagery and a reference parity set are required before use.

### MPS risk assessment

**Model math: low-to-moderate risk, verified.** Convolution, batch normalization, indexing-based exchange, interpolation, `grid_sample`, GELU, softmax, and argmax all executed on MPS through the strict-loaded extraction. Evaluation-mode SyncBatchNorm also passed a direct MPS probe.

**Unmodified Open-CD: high risk, not verified.** The full MMCV extension failed to dynamically load under torch 2.8.0, and the legacy torch 2.1.2 build failed against the current SDK. Therefore this audit does not claim unmodified `OpenCDInferencer(device='mps')` compatibility.

### CPU risk assessment

**Model math: low risk, verified.** The strict-loaded extraction completed at both tested resolutions with bounded memory and no unsupported operator.

**Unmodified Open-CD: high dependency risk, not verified end-to-end.** The framework import requires the same unusable compiled MMCV extension even though the target computation is CPU-native. Therefore this audit does not claim a production-safe full-Open-CD CPU environment.

## 10. RAM and latency planning estimate

For a future isolated single-worker extraction:

- Reserve about 1 GB for direct 512×512 CPU operation and about 2 GB for 1024×1024 CPU operation.
- Reserve roughly 1.5–2 GB of unified memory for a single 1024×1024 MPS request, allowing headroom beyond the observed driver allocation.
- Serialize MPS inference initially; concurrent workers would multiply activation and driver allocations.
- Budget approximately 0.15–0.25 seconds for a warmed 512 MPS request and 0.3–0.6 seconds for warmed 1024 MPS end-to-end processing after image decode, resize, transfers, and postprocessing. These are planning estimates derived from measured forwards, not endpoint benchmarks.
- CPU planning budgets are approximately 0.25–0.5 seconds at 512 and 0.8–1.5 seconds at 1024 after non-model overhead.

The 16 GB host has sufficient memory for one local worker at either tested size.

## 11. Recommended isolated environment

Use a new environment separate from all SatQuery environments, preferably CPython 3.11.15 already present on the host:

```text
Python 3.11.15
torch 2.8.0
numpy 1.26.4
Pillow 11.x
```

Do not include Open-CD, MMCV, MMEngine, MMSegmentation, MMDetection, or MMPreTrain in the production extraction environment unless a later task provides a separate reason. Keep the downloaded original checkpoint immutable and verify its SHA-256 at startup.

## 12. Recommended implementation strategy

1. Extract only `IA_ResNetV1c-18`, `SpatialExchange`, `ChannelExchange`, FDAF, MixFFN, and the two-class Changer head into a standalone, pure-PyTorch package.
2. Preserve checkpoint key names or provide a deterministic key mapping; require `strict=True` and reject any missing/unexpected key.
3. Reproduce BGR-to-RGB behavior, 0–255 ImageNet normalization, aspect-preserving resize, divisor-32 padding, padding removal, and final resize exactly.
4. Make device-created masks and grids explicit on the input device.
5. Use `BatchNorm2d` in eval mode for the checkpoint's SyncBN state on a single local device.
6. Return raw logits, explicit softmax probabilities, and an argmax `uint8` mask with class 1 meaning changed.
7. Before proposing production integration, run numerical parity against a trusted reference on fixed image pairs, plus LEVIR-CD samples and SatQuery-representative imagery. Set tolerances independently for CPU and MPS.
8. Keep CPU as an automatic runtime fallback for any MPS operator or allocation failure.

This strategy removes the fragile, unused compiled-op and registry stack while retaining the trained architecture and original weights.

## 13. Files and modules inspected

Open-CD v1.1.0 source:

- `configs/changer/changer_ex_r18_512x512_40k_levircd.py`
- `configs/_base_/models/changer_r18.py`
- `configs/common/standard_512x512_40k_levircd.py`
- `configs/common/standard_256x256_40k_levircd.py`
- `configs/_base_/default_runtime.py`
- `configs/_base_/datasets/levir_cd.py`
- `configs/changer/README.md`
- `docs/inference.md`
- `requirements/runtime.txt`
- `requirements/mminstall.txt`
- `setup.py`
- `opencd/__init__.py`
- `opencd/registry.py`
- `opencd/apis/opencd_inferencer.py`
- `opencd/models/__init__.py`
- `opencd/models/backbones/__init__.py`
- `opencd/models/backbones/interaction_resnet.py`
- `opencd/models/backbones/lightcdnet.py`
- `opencd/models/change_detectors/dual_input_encoder_decoder.py`
- `opencd/models/data_preprocessor.py`
- `opencd/models/decode_heads/changer.py`
- `opencd/models/utils/interaction_layer.py`
- `opencd/datasets/transforms/loading.py`
- `opencd/datasets/transforms/formatting.py`
- `opencd/datasets/transforms/transforms.py`

Installed dependency source/import paths inspected:

- MMSegmentation 1.2.2 `mmseg/models/segmentors/base.py`
- MMSegmentation 1.2.2 `mmseg/models/segmentors/encoder_decoder.py`
- MMSegmentation 1.2.2 `mmseg/apis/mmseg_inferencer.py`
- MMCV 2.0.1 compiled extension and build output

Local read-only inspection:

- `venv/pyvenv.cfg`
- `../.venv/pyvenv.cfg`
- `../sar-colorization-app-backup/venv/pyvenv.cfg`
- installed SatQuery PyTorch/torchvision metadata and MPS capability

No production source file, public API, frontend file, model checkpoint, current TTP integration, or existing virtual environment was modified.

## Preliminary feasibility classification

PROCEED_WITH_PURE_PYTORCH_EXTRACTION
