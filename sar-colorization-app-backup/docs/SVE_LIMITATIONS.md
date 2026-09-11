# SatQuery Vision Encoder v1 Limitations

- Training imagery is limited to a Lithuania summer subset of BigEarthNet v2.
- Version 1 accepts RGB or scientifically approved RGB-like optical representations, not arbitrary multispectral selections or raw SAR.
- Scene-prior similarities are cosine scores, not calibrated class probabilities.
- The encoder does not produce object boxes, masks, pixel labels, physical measurements, registration estimates, causal explanations, or ground truth.
- Bi-temporal embedding differences are high-level semantic evidence and do not localize change.
- Similarity between reference optical and SARFusionFormer-generated RGB-like imagery does not establish sensor-native equivalence, physical accuracy, or registration accuracy.
- The adapter does not contain the generic OpenCLIP backbone. A first online setup or pre-populated model cache is required.
- CPU execution is supported but the initial ViT-L/14 load has substantial latency and memory cost. MPS remains float32 and may fall back visibly to CPU.
- The process-local cache is bounded and cleared on restart; it is not a durable retrieval index.
- The recorded validation results demonstrate improvement over one generic baseline on one split. They do not establish state-of-the-art performance or direct improvement to Grounding DINO, TTP, SARFusionFormer, Pix2Pix, BLIP, or VQA.

Required disclosure:

> SatQuery Vision Encoder v1 provides scene-level embedding evidence. It does not produce pixel-level segmentation, object grounding, calibrated probabilities, or ground truth.
