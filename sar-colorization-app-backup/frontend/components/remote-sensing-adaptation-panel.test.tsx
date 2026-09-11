import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { RemoteSensingAdaptationPanel } from "@/components/remote-sensing-adaptation-panel";
import type { SVEResult } from "@/types/agent";

const result: SVEResult = {
  available: true, status: "success", model: "SatQuery Vision Encoder v1", model_version: "1.0.0",
  backbone: "OpenCLIP ViT-L-14", pretrained_weights: "laion2b_s32b_b82k", adaptation_dataset: "BigEarthNet.txt",
  adapter_checksum_fingerprint: "sha256:a99c0bf0fb44", embedding_dimension: 768, device: "mps", runtime_ms: 412,
  scene_priors: [{ label: "inland water", similarity: .31 }], caption_consistency: null, vqa_consistency: null,
  grounding_support: null, semantic_comparison: null, limitations: ["Scene-level only"], warning: null, fallback: null,
  disclaimer: "Scene-level evidence is not ground truth.",
};

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

async function render(props: { result?: SVEResult; loading?: boolean }) {
  await act(async () => { root.render(createElement(RemoteSensingAdaptationPanel, props)); });
}

describe("RemoteSensingAdaptationPanel", () => {
  it("shows provenance, readiness and textual priors accessibly", async () => {
    await render({ result });
    expect(container.querySelector("h3")?.textContent).toBe("SatQuery Vision Encoder v1");
    expect(container.textContent).toContain("Remote-sensing adapted");
    expect(container.querySelector('[role="status"]')?.textContent).toContain("Ready");
    expect(container.querySelector('[aria-label="Top scene priors"]')?.textContent).toContain("inland water");
    expect(container.querySelector("section")?.getAttribute("tabindex")).toBe("0");
  });

  it("shows loading and checksum failure without color-only meaning", async () => {
    await render({ loading: true });
    expect(container.querySelector('[role="status"]')?.textContent).toContain("Loading");
    await render({ result: { ...result, available: false, status: "checksum_failure", warning: "Artifact verification failed." } });
    expect(container.querySelector('[role="status"]')?.textContent).toContain("Checksum Failure");
    expect(container.textContent).toContain("Artifact verification failed.");
  });
});
