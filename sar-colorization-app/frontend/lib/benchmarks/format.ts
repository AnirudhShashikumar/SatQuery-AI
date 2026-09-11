import type { BenchmarkMetric, BenchmarkStatus, MetricUnit } from "./types";

export const statusMeta: Record<BenchmarkStatus, { label: string; short: string; description: string; tone: "verified" | "validation" | "warning" | "muted" | "external" }> = {
  verified_test: { label: "Verified test", short: "Test", description: "Measured on the official prescribed test split.", tone: "verified" },
  verified_validation: { label: "Verified validation", short: "Validation", description: "Measured on a known validation split; not test performance.", tone: "validation" },
  partial_validation: { label: "Partial validation", short: "Partial", description: "Measured on only part of a validation split.", tone: "warning" },
  smoke_only: { label: "Smoke only", short: "Smoke", description: "Execution or diagnostic subset evidence; not benchmark accuracy.", tone: "warning" },
  operational_only: { label: "Operational only", short: "Runtime", description: "Lifecycle, runtime, parity, or memory evidence only.", tone: "muted" },
  external_reported: { label: "Externally reported", short: "External", description: "Copied from a named external source; not reproduced by SatQuery.", tone: "external" },
  unavailable: { label: "Benchmark unavailable", short: "Unavailable", description: "No valid quality measurement has been imported.", tone: "muted" },
  not_applicable: { label: "Not applicable", short: "N/A", description: "This metric does not apply to the task.", tone: "muted" },
};

export function formatMetricValue(value: number | null, unit: MetricUnit, compact = false): string {
  if (value === null) return "Unavailable";
  if (unit === "ratio") return `${(value * 100).toFixed(compact ? 1 : 2)}%`;
  if (unit === "percentage") return `${value.toFixed(compact ? 1 : 2)}%`;
  if (unit === "milliseconds") return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(value < 100 ? 2 : 0)} ms`;
  if (unit === "megabytes") return `${value.toFixed(value < 100 ? 1 : 0)} MB`;
  if (unit === "samples_per_second") return `${value.toFixed(2)} samples/s`;
  if (unit === "db") return `${value.toFixed(2)} dB`;
  if (unit === "seconds") return `${value.toFixed(2)} s`;
  return Number.isInteger(value) ? String(value) : value.toFixed(compact ? 2 : 4);
}

export const metricValue = (metric: BenchmarkMetric) => formatMetricValue(metric.value, metric.unit);
export const formatDatasetSplit = (dataset: string | null, split: string | null) => [dataset, split].filter(Boolean).join(" · ") || "Dataset unavailable";
export const formatTimestamp = (value: string | null) => value ? new Intl.DateTimeFormat("en", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" }).format(new Date(value)) : "Not recorded";
export const shortFingerprint = (value: string | null) => value ? `${value.slice(0, 12)}…${value.slice(-6)}` : "Not recorded";
