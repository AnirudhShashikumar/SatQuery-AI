# ChangerEx pure-PyTorch standalone implementation

## Decision

The official pretrained ChangerEx IA-ResNetV1c-18 model is locally viable on
this Apple Silicon Mac as an isolated pure-PyTorch package. The official
checkpoint strict-loads without key transformation, real CPU and MPS inference
both complete, lifecycle reuse is demonstrated, and the extracted float32
outputs pass the official-source parity gate.

This work does **not** integrate ChangerEx into SatQuery production. No backend,
agent API, router, model contract, registry, TTP workflow, or frontend file was
changed.

## Selected official model

| Item | Pinned value |
|---|---|
| Model | ChangerEx |
| Backbone | IA-ResNetV1c-18 |
| Training dataset | LEVIR-CD |
| Classes | 0 `unchanged`, 1 `changed` |
| Config | `configs/changer/changer_ex_r18_512x512_40k_levircd.py` |
| Open-CD release | v1.1.0 |
| Open-CD commit | `09c03eb1077f06191c5448ea1fcf2f2f88ec98c2` |
| Checkpoint | `ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth` |
| Checkpoint size | 136,880,383 bytes |
| SHA-256 | `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618` |
| Parameters | 11,390,946 |

The official checkpoint is linked from the Open-CD Changer model table at
<https://github.com/likyoo/open-cd/tree/v1.1.0/configs/changer>.

## Config inheritance and source extraction

The selected config inherits:

1. `configs/_base_/models/changer_r18.py`
2. `configs/common/standard_512x512_40k_levircd.py`
3. `configs/common/standard_256x256_40k_levircd.py`
4. `configs/_base_/datasets/levir_cd.py` and the standard runtime/schedule bases

Only inference-critical architecture is reproduced. The source mapping is:

| Standalone module | Official source |
|---|---|
| `backbone.py` | `opencd/models/backbones/interaction_resnet.py` plus MMSeg ResNetV1c/BasicBlock |
| `interaction.py` | `opencd/models/utils/interaction_layer.py` |
| `decoder.py` | `opencd/models/decode_heads/changer.py` and `opencd/models/necks/feature_fusion.py` |
| `architecture.py` | DIEncoderDecoder's ordered six-channel split and whole-image logit resize |

Open-CD and MMSegmentation are Apache-2.0 licensed. The standalone files retain
source attribution in their module headers. No training runner, dataset,
optimizer, evaluation framework, registry system, or unrelated model is copied.

## Exact architecture

The earlier and later RGB tensors share one deep-stem ResNet-18:

- three stem convolutions: `3→32→32→64`, with strides `2,1,1`;
- official `SyncBatchNorm` and in-place ReLU after each stem convolution;
- 3×3 max-pool, stride 2;
- four two-block BasicBlock stages with channels `64,128,256,512` and stage
  strides `1,2,2,2`;
- identity interaction after stage 1, spatial exchange at period 2 after stage
  2, and channel exchange at period 2 after stages 3 and 4.

The decoder projects each temporal half of all four stages to 128 channels,
bilinearly aligns them to quarter resolution, fuses them to 64 channels, then
applies FDAF flow alignment. FDAF uses a 5×5 depthwise convolution, affine-free
InstanceNorm2d, GELU, a 1×1 four-channel flow prediction, and
`grid_sample(..., align_corners=True)`. Concatenated temporal differences pass
through the residual MixFFN and a 1×1 two-class segmentation layer. Whole-image
logits are restored with bilinear interpolation and `align_corners=False`.

There are 173 state-dict tensors/buffers and 11,390,946 trainable parameters.
No auxiliary head is present in the official checkpoint.

## Dependency strategy

Runtime dependencies are restricted to PyTorch, NumPy, Pillow, and standard
Python. CUDA is not required. Open-CD, MMCV, MMSegmentation, MMDetection, and
MMPretrain are not imported or installed by this package.

The unmodified Open-CD application stack was rejected because its eager imports
reach `mmcv.ops`; mmcv-lite cannot provide that module surface, while the locally
built full MMCV extension was ABI-incompatible with this host's PyTorch/MPS
runtime. None of those compiled operations occur in the selected ChangerEx
forward graph, so retaining that fragile framework stack would add risk without
adding inference math.

## Strict checkpoint handling

Every lifecycle load performs these steps:

1. resolve and verify that the file exists;
2. stream SHA-256 and require the exact pinned digest;
3. load the verified payload on CPU;
4. require a non-empty, tensor-only `state_dict`;
5. perform no prefix transformation (the official keys already match);
6. call `load_state_dict(..., strict=True)`;
7. fail on any missing or unexpected key;
8. verify the architecture parameter count before moving to the selected device.

Measured checkpoint report:

- top-level keys: `meta`, `optimizer`, `state_dict`;
- tensor count: 173;
- parameter count: 11,390,946;
- missing keys: none;
- unexpected keys: none;
- key transformations: none;
- checkpoint SHA: verified.

The machine-readable report is stored in each device output directory as
`checkpoint_verification.json`.

## Exact inference preprocessing

- Decode both inputs to RGB; no BGR tensor enters the standalone model.
- Require identical non-zero source dimensions and preserve caller-supplied
  earlier/later order.
- Concatenate channels as earlier RGB followed by later RGB: NCHW
  `[1, 6, H, W]`.
- Keep float32 values in `[0,255]` before normalization.
- Follow the official test pipeline's `MultiImgResize(scale=(1024,1024),
  keep_ratio=True)`: by default, scale the longest side to 1024, including
  upscaling smaller inputs, and round each target dimension with `+0.5`.
- Use bilinear half-pixel interpolation (`align_corners=False`).
- Normalize both RGB triplets with mean `[123.675,116.28,103.53]` and standard
  deviation `[58.395,57.12,57.375]`.
- Pad only the right and bottom to a multiple of 32 using zeros in normalized
  space.
- Apply no test crop and no training augmentation.

Class-1 softmax is the change-probability map. Padding is removed first. The
probability map is restored to source dimensions bilinearly; the thresholded
mask is restored with nearest-neighbor interpolation. Threshold 0.5 is the
two-class argmax boundary. No morphology is enabled. Connected components are
8-connected, and bounding boxes use exclusive `x2/y2` coordinates.

## Device and lifecycle behavior

`auto` runs a real MPS tensor operation and synchronization before selecting
MPS. If that verification fails, `auto` selects CPU and records a warning. An
explicit `mps` request fails unless MPS verifies; it falls back only when the
caller explicitly enables `allow_device_fallback`. `cpu` always forces CPU.

The process-wide lifecycle is lazy and protected by a reentrant lock. It tracks
`unloaded`, `loading`, `ready`, and `failed`; device, verification, load/reuse
and inference counts, load/last-inference time, parameter count, measured peak
memory, warnings, and a bounded single-line safe error. A failed lifecycle is
sticky. Checkpoint and requested device are immutable after initial process
configuration. All model execution uses `torch.inference_mode()` and float32.

## Official-source parity

An independent CPU reference was instantiated from the unmodified pinned
Open-CD Changer/interaction modules and the official MMSeg ResNet source. A
minimal inference-only framework shim supplied registry-independent
BaseDecodeHead plumbing and suppressed unrelated eager custom-op imports; it did
not use any extracted architecture code. The official checkpoint strict-loaded
into that reference with zero missing/unexpected keys.

The NASA/USGS Hanford pair was transformed independently according to the
pinned official test configuration. The preprocessed tensor was
`[1,6,768,1024]`, SHA-256
`4891c980f62096fa1b8ee4c95f4be3090b90ce33de7aec835a76453f369b7311`.
The standalone package produced the same tensor hash.

Acceptance thresholds were:

- probability maximum absolute difference ≤ `1e-4`;
- probability mean absolute difference ≤ `1e-5`;
- binary agreement ≥ `0.999`;
- mask IoU ≥ `0.998`;
- changed-percentage difference ≤ `0.1` percentage point.

Measured extracted-versus-official result:

| Metric | Result |
|---|---:|
| Logits shape | `[1,2,768,1024]` both |
| Logits max absolute difference | 0.0 |
| Logits mean absolute difference | 0.0 |
| Restored probability max absolute difference | 0.0 |
| Restored probability mean absolute difference | 0.0 |
| Binary mask agreement | 1.0 |
| Mask IoU | 1.0 |
| Changed-percentage difference | 0.0 pp |
| Acceptance | pass |

This is full float32 output parity for the measured pair, not merely mask-level
parity. The reference artifact, source-file hashes, extracted arrays, thresholds,
and comparison are under `artifacts/changerex_local_smoke/parity/`.

## Real operational smoke

The real pair is the NASA Earth Observatory Landsat 7 view of the Hanford,
Washington brushfire: May 6, 2000 and July 9, 2000, credited on the source page
to Ron Beck, USGS EROS Data Center. Original files are 2400×1801. Their URLs,
credit, hashes, and use basis are recorded in `source_manifest.json`.

This pair does not include a pixel-level ground-truth mask. The smoke proves
runtime/output behavior only and makes no accuracy claim.

| Measure | MPS | CPU |
|---|---:|---:|
| Inference completed | yes | yes |
| Load count | 1 | 1 |
| Reuse count | 6 | 6 |
| Load time | 0.2490 s | 0.1019 s |
| First end-to-end time | 0.7625 s | 0.5580 s |
| First model-forward time | 0.3929 s | 0.3677 s |
| Warm mean, 5 forwards | 0.0960 s | 0.3164 s |
| Warm median | 0.0952 s | 0.3139 s |
| Warm P95 | 0.0981 s | 0.3245 s |
| Peak measured memory | 1121.0 MB driver allocation | 1126.7 MB process RSS |
| Repeat max probability difference | 0.0 | 0.0 |
| Source-sized output | 2400×1801 | 2400×1801 |
| Probability range | `1.2882e-17–0.9847781` | `1.2883e-17–0.9847782` |
| Changed percentage | 0.163682% | 0.163682% |
| Regions | 5 | 5 |

MPS versus CPU maximum/mean probability differences were `2.95043e-06` and
`5.39955e-09`; masks were identical (agreement 1.0, IoU 1.0). Both repeated
runs were deterministic. These timings are specific to the 1024×768 processed
input on this Apple M5 host and should not be generalized to every Mac.

## Test and non-regression result

- ChangerEx focused tests: 22 passed, including one real official checkpoint
  strict-load test outside mocks.
- Full repository suite: 495 passed, 4 skipped.
- Full-suite warnings: 27 pre-existing environment/Pillow warnings; no test
  failures.
- Byte-hash guards passed for protected backend/API/router/contracts,
  `change_analysis.py`, `ttp_change.py`, the frontend source tree, and the TTP
  service Python tree.
- Package compilation passed with an isolated bytecode cache.

## Artifacts and CLI

The module CLI is:

```bash
python -m changerex_local \
  --earlier before.png \
  --later after.png \
  --checkpoint ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth \
  --output-dir output \
  --device auto \
  --threshold 0.5
```

Each output contains `probability_map.npy`, a 16-bit probability PNG, binary
and display masks, an overlay that retains the later image, region JSON, result
JSON, checkpoint verification, environment diagnostics, and a benchmark report.
Errors return bounded JSON on stderr with a non-zero status.

## Limitations

- Parity is exact for the measured pair and runtime; it is not a proof over all
  possible tensors or future PyTorch releases.
- The parity reference required a narrow import shim because unmodified Open-CD
  eagerly imports unrelated MMCV custom ops. That reference is for validation,
  not a supported runtime.
- Inputs must be co-registered, equally sized RGB images. Misregistration,
  illumination, seasonal, and sensor differences can appear as change.
- LEVIR-CD is primarily a building-change dataset. No accuracy or calibration
  claim is made for the Hanford fire imagery or other domains.
- Peak CPU RSS includes interpreter/runtime allocations; MPS reports driver
  allocation. The figures are useful operational bounds, not directly
  equivalent allocator metrics.

## Recommended future production integration contract

Keep this package unchanged and introduce a thin adapter only in a future,
separately approved production task. Configure one checkpoint/device lifecycle
at process startup or service startup, expose its state through existing health
conventions, preserve the ordered earlier/later contract, and translate the
standalone result into existing production schemas without changing this model's
preprocessing or outputs. Retain the current TTP path until product routing and
fallback behavior are explicitly designed and tested.

CHANGEREX_STANDALONE_READY
