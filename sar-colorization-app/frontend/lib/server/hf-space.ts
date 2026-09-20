import "server-only";

import { createHmac, timingSafeEqual } from "node:crypto";
import { Client } from "@gradio/client";

const DEFAULT_SPACE_ID = "AnirudhShashikumar/SatQuery-AI";
const DEFAULT_QUERY_TIMEOUT_MS = 240_000;
const DEFAULT_HEALTH_TIMEOUT_MS = 30_000;

export class SpaceProxyError extends Error {
  constructor(public readonly status: number, public readonly code: string, message: string) {
    super(message);
    this.name = "SpaceProxyError";
  }
}

export function spaceId() {
  return process.env.HF_SPACE_ID?.trim() || DEFAULT_SPACE_ID;
}

export function spaceOrigin() {
  const slug = spaceId().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  if (!slug) throw new SpaceProxyError(503, "PROXY_MISCONFIGURED", "The Space identifier is invalid.");
  return `https://${slug}.hf.space`;
}

function timeoutMs(kind: "query" | "health") {
  const fallback = kind === "query" ? DEFAULT_QUERY_TIMEOUT_MS : DEFAULT_HEALTH_TIMEOUT_MS;
  const configured = Number.parseInt(process.env.HF_REQUEST_TIMEOUT_MS ?? "", 10);
  if (!Number.isFinite(configured)) return fallback;
  return Math.min(295_000, Math.max(kind === "query" ? 90_000 : 10_000, configured));
}

function tokenOption() {
  const token = process.env.HF_TOKEN?.trim();
  if (!token) return undefined;
  if (!token.startsWith("hf_")) {
    throw new SpaceProxyError(503, "PROXY_MISCONFIGURED", "The server-side Hugging Face token is invalid.");
  }
  return { token: token as `hf_${string}` };
}

function withTimeout<T>(operation: Promise<T>, milliseconds: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new SpaceProxyError(504, "SPACE_TIMEOUT", "The ZeroGPU request timed out.")),
      milliseconds,
    );
    operation.then(
      value => { clearTimeout(timer); resolve(value); },
      error => { clearTimeout(timer); reject(error); },
    );
  });
}

export async function callSpace(endpoint: string, payload: Record<string, unknown>, kind: "query" | "health" = "query") {
  let client: Client | undefined;
  try {
    client = await withTimeout(Client.connect(spaceId(), tokenOption()), timeoutMs(kind));
    const result = await withTimeout(client.predict<unknown[]>(endpoint, payload), timeoutMs(kind));
    if (!Array.isArray(result.data)) {
      throw new SpaceProxyError(502, "INVALID_SPACE_RESPONSE", "The Space returned an invalid response.");
    }
    return result.data;
  } catch (error) {
    if (error instanceof SpaceProxyError) throw error;
    const message = error instanceof Error ? error.message.toLowerCase() : "";
    if (message.includes("sleep") || message.includes("start") || message.includes("build") || message.includes("queue")) {
      throw new SpaceProxyError(503, "SPACE_UNAVAILABLE", "The ZeroGPU Space is not ready yet.");
    }
    throw new SpaceProxyError(502, "SPACE_REQUEST_FAILED", "The Hugging Face Space request failed.");
  } finally {
    client?.close();
  }
}

function artifactSecret() {
  const secret = process.env.HF_ARTIFACT_PROXY_SECRET?.trim();
  if (!secret || secret.length < 32) {
    throw new SpaceProxyError(
      503,
      "PROXY_MISCONFIGURED",
      "The server-side artifact proxy secret is not configured.",
    );
  }
  return secret;
}

function sanitizedArtifactUrl(remoteValue: string) {
  let url: URL;
  try {
    url = new URL(remoteValue, `${spaceOrigin()}/`);
  } catch {
    throw new SpaceProxyError(502, "INVALID_ARTIFACT_URL", "The Space returned an invalid artifact URL.");
  }
  const expected = new URL(spaceOrigin());
  const allowedPath = ["/gradio_api/file=", "/gradio_api/file/", "/file=", "/file/"]
    .some(prefix => url.pathname.startsWith(prefix));
  if (url.protocol !== "https:" || url.origin !== expected.origin || !allowedPath) {
    throw new SpaceProxyError(502, "INVALID_ARTIFACT_URL", "The Space returned an untrusted artifact URL.");
  }
  url.searchParams.delete("__sign");
  url.searchParams.delete("token");
  url.searchParams.delete("access_token");
  return url.toString();
}

function signature(source: string) {
  return createHmac("sha256", artifactSecret()).update(source).digest("base64url");
}

export function artifactProxyUrl(remoteValue: string) {
  const source = Buffer.from(sanitizedArtifactUrl(remoteValue), "utf8").toString("base64url");
  return `/api/agent/artifacts?source=${encodeURIComponent(source)}&signature=${encodeURIComponent(signature(source))}`;
}

export function verifiedArtifactUrl(source: string | null, providedSignature: string | null) {
  if (!source || !providedSignature) {
    throw new SpaceProxyError(400, "INVALID_ARTIFACT_REQUEST", "The artifact request is incomplete.");
  }
  const expected = Buffer.from(signature(source));
  const provided = Buffer.from(providedSignature);
  if (expected.length !== provided.length || !timingSafeEqual(expected, provided)) {
    throw new SpaceProxyError(403, "INVALID_ARTIFACT_SIGNATURE", "The artifact request is not authorized.");
  }
  try {
    return sanitizedArtifactUrl(Buffer.from(source, "base64url").toString("utf8"));
  } catch (error) {
    if (error instanceof SpaceProxyError) throw error;
    throw new SpaceProxyError(400, "INVALID_ARTIFACT_REQUEST", "The artifact request is invalid.");
  }
}

export function spaceAuthorizationHeaders(): Record<string, string> {
  const token = process.env.HF_TOKEN?.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export function proxyErrorBody(error: SpaceProxyError) {
  return { detail: { code: error.code, message: error.message } };
}
