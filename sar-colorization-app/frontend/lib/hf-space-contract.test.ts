import { describe, expect, it } from "vitest";
import { buildSpaceCall, rewriteArtifactReferences, SPACE_ENDPOINTS } from "@/lib/hf-space-contract";

function baseForm(mode: "single" | "cross_modal" | "bi_temporal") {
  const form = new FormData();
  form.set("input_mode", mode);
  form.set("query", "Analyze this observation.");
  form.set("primary_modality", "optical");
  form.set("primary_image_modality", "optical_rgb");
  form.set("primary_image", new File(["primary"], "primary.png", { type: "image/png" }));
  form.set("use_cache", "true");
  form.set("force_rerun", "false");
  return form;
}

describe("Hugging Face Space compatibility contract", () => {
  it("maps the existing single-image multipart request", () => {
    const call = buildSpaceCall(baseForm("single"));
    expect(call.endpoint).toBe(SPACE_ENDPOINTS.single);
    expect(call.payload).toMatchObject({
      query: "Analyze this observation.",
      primary_modality: "optical",
      primary_image_modality: "optical_rgb",
      use_cache: true,
      force_rerun: false,
    });
    expect(call.payload.primary_image).toBeInstanceOf(Blob);
  });

  it("maps the existing Optical+SAR multipart request with fixed observation roles", () => {
    const form = baseForm("cross_modal");
    form.set("secondary_image", new File(["sar"], "sar.tif", { type: "image/tiff" }));
    form.set("secondary_modality", "sar");
    const call = buildSpaceCall(form);
    expect(call.endpoint).toBe(SPACE_ENDPOINTS.cross_modal);
    expect(call.payload.optical_image).toBeInstanceOf(Blob);
    expect(call.payload.sar_image).toBeInstanceOf(Blob);
    expect(call.payload).not.toHaveProperty("primary_modality");
  });

  it("maps the existing bi-temporal multipart request", () => {
    const form = baseForm("bi_temporal");
    form.set("secondary_image", new File(["later"], "later.png", { type: "image/png" }));
    form.set("primary_date", "2025-01-01");
    form.set("secondary_date", "2025-02-01");
    form.set("force_rerun", "true");
    const call = buildSpaceCall(form);
    expect(call.endpoint).toBe(SPACE_ENDPOINTS.bi_temporal);
    expect(call.payload).toMatchObject({
      query: "Analyze this observation.",
      primary_date: "2025-01-01",
      secondary_date: "2025-02-01",
      primary_modality: "optical",
      use_cache: true,
      force_rerun: true,
    });
  });

  it("rewrites every manifest-backed evidence reference without dropping evidence", () => {
    const response = {
      primary_image_metadata: { preview_url: "/api/agent/previews/input.png" },
      evidence: [{ reference: "/api/agent/previews/evidence.png" }],
    };
    const manifest = [
      { index: 0, reference: "/api/agent/previews/input.png", filename: "input.png" },
      { index: 1, reference: "/api/agent/previews/evidence.png", filename: "evidence.png" },
    ];
    const files = [
      { url: "https://example.hf.space/gradio_api/file=input.png" },
      { url: "https://example.hf.space/gradio_api/file=evidence.png" },
    ];
    const rewritten = rewriteArtifactReferences(response, manifest, files, url => `/proxy?url=${url}`);
    expect(rewritten).toEqual({
      primary_image_metadata: { preview_url: "/proxy?url=https://example.hf.space/gradio_api/file=input.png" },
      evidence: [{ reference: "/proxy?url=https://example.hf.space/gradio_api/file=evidence.png" }],
    });
  });
});
