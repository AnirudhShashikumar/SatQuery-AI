import { benchmarkStatuses, type BenchmarkMetric, type BenchmarkResult, type BenchmarkSuite, type DemoManifest, type MetricUnit } from "./types";

const metricUnits = new Set<MetricUnit>(["ratio", "percentage", "milliseconds", "seconds", "megabytes", "count", "samples_per_second", "db", "score"]);
const statuses = new Set<string>(benchmarkStatuses);
const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const isNullableString = (value: unknown): value is string | null => value === null || typeof value === "string";
const isNullableNumber = (value: unknown): value is number | null => value === null || typeof value === "number" && Number.isFinite(value);
const isStringArray = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === "string");

function isMetric(value: unknown): value is BenchmarkMetric {
  if (!isObject(value)) return false;
  return typeof value.id === "string" && typeof value.label === "string" && isNullableNumber(value.value)
    && typeof value.unit === "string" && metricUnits.has(value.unit as MetricUnit)
    && (typeof value.higher_is_better === "boolean" || value.higher_is_better === null)
    && typeof value.description === "string" && typeof value.primary === "boolean";
}

export function validateBenchmarkRecord(value: unknown): { valid: true; value: BenchmarkResult } | { valid: false; errors: string[] } {
  const errors: string[] = [];
  if (!isObject(value)) return { valid: false, errors: ["record must be an object"] };
  if (value.schema_version !== "1.0.0") errors.push("unsupported schema_version");
  for (const key of ["benchmark_id", "specialist_id", "display_name", "task"] as const) if (typeof value[key] !== "string" || !value[key]) errors.push(`${key} is required`);
  if (!isObject(value.model)) errors.push("model is required");
  else {
    for (const key of ["name", "version", "architecture"] as const) if (typeof value.model[key] !== "string" || !value.model[key]) errors.push(`model.${key} is required`);
    if (!isNullableString(value.model.checkpoint)) errors.push("model.checkpoint must be string or null");
    if (!(value.model.checkpoint_sha256 === null || typeof value.model.checkpoint_sha256 === "string" && /^[a-f0-9]{64}$/.test(value.model.checkpoint_sha256))) errors.push("model.checkpoint_sha256 is invalid");
  }
  if (!isObject(value.evaluation)) errors.push("evaluation is required");
  else {
    if (typeof value.evaluation.status !== "string" || !statuses.has(value.evaluation.status)) errors.push("evaluation.status is invalid");
    if (!(value.evaluation.sample_count === null || Number.isInteger(value.evaluation.sample_count) && Number(value.evaluation.sample_count) >= 0)) errors.push("evaluation.sample_count is invalid");
    if (typeof value.evaluation.protocol !== "string") errors.push("evaluation.protocol is required");
    if (!isObject(value.evaluation.environment)) errors.push("evaluation.environment is required");
  }
  if (!Array.isArray(value.metrics) || !value.metrics.every(isMetric)) errors.push("metrics are invalid");
  if (!isObject(value.performance)) errors.push("performance is required");
  if (!isObject(value.artifacts)) errors.push("artifacts are required");
  if (!isStringArray(value.limitations) || !isStringArray(value.notes)) errors.push("limitations and notes must be arrays");
  if (!isObject(value.provenance) || !isStringArray(value.provenance.source_artifacts)) errors.push("provenance is invalid");
  else if (value.provenance.source_artifacts.some(source => source.startsWith("/") || source.includes("/Users/") || source.includes("/home/"))) errors.push("provenance exposes a private path");
  return errors.length ? { valid: false, errors } : { valid: true, value: value as BenchmarkResult };
}

export function validateBenchmarkSuite(value: unknown): { valid: true; value: BenchmarkSuite } | { valid: false; errors: string[] } {
  if (!isObject(value)) return { valid: false, errors: ["suite must be an object"] };
  const errors: string[] = [];
  if (value.schema_version !== "1.0.0") errors.push("unsupported suite schema");
  if (typeof value.suite_version !== "string" || typeof value.generated_at !== "string") errors.push("suite metadata is incomplete");
  if (!isStringArray(value.specialists) || !isStringArray(value.benchmark_record_ids)) errors.push("suite record lists are invalid");
  if (!isObject(value.preferred_record_by_specialist) || !Object.values(value.preferred_record_by_specialist).every(item => typeof item === "string")) errors.push("preferred record map is invalid");
  return errors.length ? { valid: false, errors } : { valid: true, value: value as BenchmarkSuite };
}

export function validateDemoManifest(value: unknown): { valid: true; value: DemoManifest } | { valid: false; errors: string[] } {
  if (!isObject(value) || value.schema_version !== "1.0.0" || !Array.isArray(value.cases)) return { valid: false, errors: ["demo manifest is invalid"] };
  const errors: string[] = [];
  for (const item of value.cases) {
    if (!isObject(item) || typeof item.demo_id !== "string" || typeof item.title !== "string") errors.push("demo identity is invalid");
    else if (!String(item.primary_image).startsWith("/demos/") || !String(item.attribution || "") || !String(item.license || "")) errors.push(`demo ${item.demo_id} lacks a safe asset or attribution`);
  }
  return errors.length ? { valid: false, errors } : { valid: true, value: value as DemoManifest };
}
