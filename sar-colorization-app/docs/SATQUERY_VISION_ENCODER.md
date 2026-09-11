# SatQuery Vision Encoder v1

SatQuery Vision Encoder v1 is a scene-level, remote-sensing-adapted vision-language encoder. It reconstructs OpenCLIP `ViT-L-14` with pretrained weights `laion2b_s32b_b82k`, then applies a verified adapter containing only the parameters adapted during training. The normalized embedding dimension is 768.

## Training provenance

- Text: BigEarthNet.txt captions
- Imagery: BigEarthNet v2, Lithuania summer subset
- RGB bands: Sentinel-2 B04, B03, B02
- Train / validation / test pairs: 4,008 / 2,291 / 2,053
- Training: 5 epochs, BF16, OpenCLIP contrastive objective
- Adapted parameters: final two visual transformer blocks, visual post-layer norm and projection, and logit scale

The geographic and seasonal scope is narrow. Retrieval metrics improved over the recorded generic OpenCLIP baseline on the validation split, but absolute retrieval remains low. This is adaptation evidence, not a broad performance claim.

## Artifact integrity

The runtime requires five files under `models/satquery_vision_encoder_v1/`: the adapter, model card, preprocessing metadata, validation comparison, and checksum manifest. The adapter must match SHA-256 `a99c0bf0fb44044988ef1698483888c8a2e3a047d2d2d56478837575cf7626ea`.

Verification occurs before OpenCLIP construction. The loader validates model version, backbone, pretrained tag, embedding dimension, preprocessing contract, evaluation metadata, adapter namespace, the exact 28-key adapted parameter set, and every tensor shape. It loads weights with `torch.load(..., weights_only=True)`, sets evaluation mode, disables gradients, and does not expose the model or embeddings through standard APIs.

## Supported uses

Supported uses are normalized EO image/text embeddings, image-text similarity, controlled scene priors, caption consistency and candidate reranking, bounded retrieval by opaque evidence ID, advisory routing support, VQA evidence consistency, grounding-environment plausibility, generated-RGB semantic support, and scene-level bi-temporal comparison.

SatQuery Vision Encoder v1 provides scene-level embedding evidence. It does not produce pixel-level segmentation, object grounding, calibrated probabilities, or ground truth.
