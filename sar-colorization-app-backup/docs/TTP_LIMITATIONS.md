# TTP limitations and scientific disclosure

TTP predicts a binary `unchanged`/`changed` mask. It does not identify flooding, construction, demolition, deforestation, crop transitions, damage, or any causal explanation. The UI and reports never infer a semantic change type from TTP alone.

The official checkpoint was trained on LEVIR-CD building-change imagery. Generalization may be weaker for vegetation, flooding, seasonal variation, unfamiliar sensors, SAR, non-urban scenes, different spatial resolutions, compression artifacts, or radiometric differences. TTP is not routed to SAR or cross-modal pairs.

Equal pixel dimensions do not prove geographic correspondence for ordinary PNG/JPEG previews. GeoVision discloses visual-only compatibility and does no automatic registration. Operational users must validate alignment and sensor suitability before relying on a learned mask.

Mask IoU and pixel agreement compare two methods; they are not ground-truth accuracy. Evidence-consistency labels are operational disclosures and are not calibrated probabilities. TTP and deterministic masks may share correlated sensitivity to alignment, illumination, season, or source quality.

Raw TTP output is preserved as a strict binary audit artifact. Display masks and overlays are separate visualizations. No aggressive small-object removal is applied to the learned mask without benchmark justification.

Required wording: “TTP output is a model-generated binary change prediction and is not ground truth.”
