# TTP integration architecture

GeoVision uses the official [KyanChen/TTP](https://github.com/KyanChen/TTP) implementation pinned to commit `431377d` and the Hugging Face `KyanChen/TTP` checkpoint `epoch_260.pth`. The required SHA-256 is `60294429b3d22310e1451b1059b953b44323adfe3af4eae1d75ae1709e610cb9`.

For a validated, pixel-compatible RGB optical pair, GeoVision first runs its unchanged deterministic normalized-difference analyzer. If `TTP_ENABLED=true`, the configured mode is `hybrid` or `ttp`, and the persistent service reports a verified ready CUDA lifecycle, the exact validated source bytes are sent to TTP. GeoVision validates the returned mask dimensions, binary values, statistics, model ID, checkpoint fingerprint, content type, and response size.

In hybrid mode the TTP learned mask is primary. The deterministic difference map, mask, and overlay remain independent supporting evidence. GeoVision computes intersection, union, mask IoU, pixel agreement/disagreement, changed-class agreement, and background agreement without averaging either mask. These are evidence-consistency measures, not accuracy against ground truth.

TTP is never called for single images, SAR–SAR, optical–SAR, mismatched dimensions, incompatible pairs, unsupported band layouts, or pairs requiring alignment. GeoVision never registers, reprojects, warps, resizes, or silently aligns scientific imagery. Any timeout, unavailable service, CUDA OOM, malformed response, or invalid artifact produces a typed, visible deterministic fallback.

Optional response fields are `change_engine`, `ttp_result`, `mask_comparison`, `evidence_consistency`, `deterministic_statistics`, and source-labelled preview URLs. Existing deterministic fields remain valid.

Cache identity includes both content hashes, task/query family, dates, requested engine mode, enabled state, TTP model ID, checkpoint fingerprint, deterministic analyzer version, modality, and controlled parameters. Deterministic-only and hybrid results cannot share a cache entry.

TTP output is a model-generated binary change prediction and is not ground truth.
