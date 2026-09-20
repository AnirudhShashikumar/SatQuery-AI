export const SPACE_ENDPOINTS = {
  single: "/single_image",
  cross_modal: "/optical_sar",
  bi_temporal: "/bitemporal",
} as const;

export type SpaceInputMode = keyof typeof SPACE_ENDPOINTS;

export type SpaceCall = {
  endpoint: (typeof SPACE_ENDPOINTS)[SpaceInputMode];
  payload: Record<string, unknown>;
};

export class SpaceContractError extends Error {
  constructor(message: string, public readonly status = 400) {
    super(message);
    this.name = "SpaceContractError";
  }
}

const MODALITIES = new Set(["optical", "multispectral", "sar", "unknown"]);
const IMAGE_MODALITIES = new Set([
  "auto",
  "optical_rgb",
  "optical_grayscale",
  "panchromatic",
  "sar_preview",
  "sar_vv",
  "sar_vh",
  "sar_vv_vh",
  "multispectral",
  "unknown",
]);

function text(form: FormData, name: string, fallback = "") {
  const value = form.get(name);
  return typeof value === "string" ? value.trim() : fallback;
}

function choice(form: FormData, name: string, allowed: Set<string>, fallback: string) {
  const value = text(form, name, fallback);
  if (!allowed.has(value)) throw new SpaceContractError(`Unsupported ${name}.`, 422);
  return value;
}

function bool(form: FormData, name: string, fallback: boolean) {
  const value = text(form, name);
  if (!value) return fallback;
  if (value === "true") return true;
  if (value === "false") return false;
  throw new SpaceContractError(`Invalid ${name}.`, 422);
}

function upload(form: FormData, name: string) {
  const value = form.get(name);
  if (!(value instanceof Blob) || value.size === 0) {
    throw new SpaceContractError(`Missing ${name}.`, 422);
  }
  return value;
}

export function buildSpaceCall(form: FormData): SpaceCall {
  const mode = text(form, "input_mode") as SpaceInputMode;
  if (!(mode in SPACE_ENDPOINTS)) throw new SpaceContractError("Unsupported input_mode.", 422);

  const query = text(form, "query");
  if (!query) throw new SpaceContractError("A query is required.", 422);

  const primaryImage = upload(form, "primary_image");
  const useCache = bool(form, "use_cache", true);
  const forceRerun = bool(form, "force_rerun", false);

  if (mode === "single") {
    return {
      endpoint: SPACE_ENDPOINTS.single,
      payload: {
        primary_image: primaryImage,
        query,
        primary_modality: choice(form, "primary_modality", MODALITIES, "optical"),
        primary_image_modality: choice(form, "primary_image_modality", IMAGE_MODALITIES, "auto"),
        use_cache: useCache,
        force_rerun: forceRerun,
      },
    };
  }

  const secondaryImage = upload(form, "secondary_image");
  if (mode === "cross_modal") {
    return {
      endpoint: SPACE_ENDPOINTS.cross_modal,
      payload: {
        optical_image: primaryImage,
        sar_image: secondaryImage,
        query,
        use_cache: useCache,
        force_rerun: forceRerun,
      },
    };
  }

  const primaryDate = text(form, "primary_date");
  const secondaryDate = text(form, "secondary_date");
  if (!primaryDate || !secondaryDate) {
    throw new SpaceContractError("Both bi-temporal dates are required.", 422);
  }
  return {
    endpoint: SPACE_ENDPOINTS.bi_temporal,
    payload: {
      earlier_image: primaryImage,
      later_image: secondaryImage,
      query,
      primary_date: primaryDate,
      secondary_date: secondaryDate,
      primary_modality: choice(form, "primary_modality", MODALITIES, "optical"),
      use_cache: useCache,
      force_rerun: forceRerun,
    },
  };
}

type ArtifactManifestEntry = {
  index: number;
  reference: string;
};

function manifestEntries(value: unknown): ArtifactManifestEntry[] {
  if (!Array.isArray(value)) throw new SpaceContractError("Space artifact manifest is invalid.", 502);
  return value.map(item => {
    if (
      !item ||
      typeof item !== "object" ||
      !Number.isInteger((item as { index?: unknown }).index) ||
      typeof (item as { reference?: unknown }).reference !== "string"
    ) {
      throw new SpaceContractError("Space artifact manifest is invalid.", 502);
    }
    return item as ArtifactManifestEntry;
  });
}

function fileUrl(value: unknown): string {
  if (typeof value === "string" && value) return value;
  if (value && typeof value === "object" && typeof (value as { url?: unknown }).url === "string") {
    return (value as { url: string }).url;
  }
  throw new SpaceContractError("Space evidence file is unavailable.", 502);
}

function replaceReferences(value: unknown, replacements: ReadonlyMap<string, string>): unknown {
  if (typeof value === "string") return replacements.get(value) ?? value;
  if (Array.isArray(value)) return value.map(item => replaceReferences(item, replacements));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, replaceReferences(item, replacements)]),
    );
  }
  return value;
}

export function rewriteArtifactReferences(
  response: unknown,
  manifest: unknown,
  files: unknown,
  proxyUrl: (remoteUrl: string) => string,
) {
  const entries = manifestEntries(manifest);
  if (!entries.length) return response;
  if (!Array.isArray(files)) throw new SpaceContractError("Space evidence files are unavailable.", 502);

  const replacements = new Map<string, string>();
  for (const entry of entries) {
    if (entry.index < 0 || entry.index >= files.length) {
      throw new SpaceContractError("Space evidence file mapping is incomplete.", 502);
    }
    replacements.set(entry.reference, proxyUrl(fileUrl(files[entry.index])));
  }
  return replaceReferences(response, replacements);
}

export function parseJsonOutput(value: unknown): unknown {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return value;
  }
}
