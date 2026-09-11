export type ModelDefinition = {
  slug: string;
  name: string;
  role: string;
  summary: string;
  integration: string;
  checkpoint: string;
  inputs: string[];
  output: string;
  provenance: string;
  limitations: string[];
  actionHref: string;
  actionLabel: string;
  modelHealthKey?: string;
  specialistHealthKey?: string;
  supportingStack?: string[];
};

export const modelCatalog = {
  "vision-encoder": {
    slug: "vision-encoder",
    name: "SatQuery Vision Encoder",
    role: "Remote-sensing representation and evidence support",
    summary: "Supplies remote-sensing-aware visual evidence to routed captioning and question-answering workflows. Runtime readiness is reported by the Agent specialist registry.",
    integration: "Assistant specialist",
    checkpoint: "Reported by the specialist at inference time",
    inputs: ["Optical RGB", "Multispectral display composites"],
    output: "Scene embeddings and supporting evidence",
    provenance: "SatQuery specialist registry · satquery_vision_encoder_v1",
    limitations: ["Eligibility depends on validated modality and band layout.", "Representation evidence does not itself constitute a calibrated answer."],
    actionHref: "/assistant?mode=single",
    actionLabel: "Open Single Image",
    specialistHealthKey: "satquery_vision_encoder_v1",
  },
  pix2pix: {
    slug: "pix2pix",
    name: "Pix2Pix",
    role: "Visual-quality SAR-to-optical reconstruction",
    summary: "Produces an optical-style reconstruction from SAR imagery through the existing local Pix2Pix inference workflow.",
    integration: "Dedicated reconstruction workflow",
    checkpoint: "pix2pix_gen_180.pth",
    inputs: ["SAR PNG or JPEG", "SAR TIFF", "Optional optical ground truth"],
    output: "RGB optical-style reconstruction",
    provenance: "Repository Pix2Pix generator checkpoint",
    limitations: ["Generated detail may be visually plausible rather than physically exact.", "PSNR and SSIM are available only when ground truth is supplied."],
    actionHref: "/pix2pix",
    actionLabel: "Run Pix2Pix",
    modelHealthKey: "pix2pix",
  },
  sarfusionformer: {
    slug: "sarfusionformer",
    name: "SARFusionFormer",
    role: "Structure-preserving SAR-to-optical reconstruction",
    summary: "Processes combined or paired VV/VH SAR inputs while preserving raw model output separately from display enhancement.",
    integration: "Dedicated reconstruction workflow",
    checkpoint: "sarfusionformer_256_decoder_best.pt",
    inputs: ["Combined dual-channel SAR", "Separate VV and VH rasters", "Optional optical ground truth"],
    output: "Raw radiometric and display-enhanced RGB products",
    provenance: "Repository SARFusionFormer decoder checkpoint",
    limitations: ["Enhanced output is visualization-only and is excluded from scientific metrics.", "Input channel layout must be detected or supplied correctly."],
    actionHref: "/structure",
    actionLabel: "Run SARFusionFormer",
    modelHealthKey: "sarfusionformer",
  },
  "change-detection": {
    slug: "change-detection",
    name: "Change Detection",
    role: "Hybrid bi-temporal change evidence",
    summary: "Combines the TTP learned binary change detector with an independent deterministic analyzer. The learned mask is primary when the CUDA service is ready; deterministic evidence remains available for comparison and fallback.",
    integration: "Bi-temporal Assistant workflow",
    checkpoint: "TTP epoch_260.pth · fingerprint disclosed by service",
    inputs: ["Co-registered before image", "Co-registered after image", "Distinct observation dates"],
    output: "Binary masks, overlays, changed-area statistics, and controlled answers",
    provenance: "TTP SAM ViT-L + LoRA SiamEncoderDecoder · LEVIR-CD; deterministic normalized-difference analyzer",
    limitations: ["TTP is optimized primarily for optical building-change imagery.", "Outputs are binary change predictions, not ground truth.", "The workflow does not infer the semantic cause of change."],
    actionHref: "/assistant?mode=bi_temporal",
    actionLabel: "Run Bi-temporal Analysis",
    specialistHealthKey: "ttp_change_detector",
    supportingStack: ["TTP learned change detector", "Deterministic normalized-difference analyzer", "Explicit hybrid agreement and fallback"],
  },
  "color-corrector": {
    slug: "color-corrector",
    name: "Color Corrector",
    role: "Optional post-reconstruction display refinement",
    summary: "Applies the separately loaded residual colour-correction network only when requested in the SARFusionFormer workflow. Raw reconstruction output remains preserved.",
    integration: "Optional SARFusionFormer stage",
    checkpoint: "color_corrector_256_best.pt",
    inputs: ["SARFusionFormer RGB reconstruction"],
    output: "Bounded residual RGB correction",
    provenance: "Repository ColorCorrectionNet checkpoint, loaded independently",
    limitations: ["It is not a replacement for the raw model output.", "Corrected metrics require compatible optical ground truth.", "Colour refinement does not establish scientific fidelity."],
    actionHref: "/structure",
    actionLabel: "Open Color Correction Workflow",
    modelHealthKey: "color_corrector",
  },
} as const satisfies Record<string, ModelDefinition>;

export type ModelSlug = keyof typeof modelCatalog;

export function isModelSlug(value: string): value is ModelSlug {
  return value in modelCatalog;
}
