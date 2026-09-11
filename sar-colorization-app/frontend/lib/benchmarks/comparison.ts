import type { BenchmarkMetric, BenchmarkResult } from "./types";

export type ComparisonCompatibility = { compatible: boolean; reason: string; sharedMetricIds: string[] };

export function comparisonCompatibility(a: BenchmarkResult, b: BenchmarkResult): ComparisonCompatibility {
  if (a.task !== b.task) return { compatible: false, reason: "Quality metrics cannot be ranked across different tasks.", sharedMetricIds: [] };
  if (a.evaluation.dataset !== b.evaluation.dataset || a.evaluation.split !== b.evaluation.split) return { compatible: false, reason: "Runs must use the same dataset and split.", sharedMetricIds: [] };
  const denied = new Set(["unavailable", "operational_only", "not_applicable"]);
  if (denied.has(a.evaluation.status) || denied.has(b.evaluation.status)) return { compatible: false, reason: "Operational or unavailable records do not support quality comparison.", sharedMetricIds: [] };
  const bMetrics = new Map(b.metrics.map(metric => [metric.id, metric]));
  const sharedMetricIds = a.metrics.filter(metric => metric.value !== null && bMetrics.get(metric.id)?.value !== null && bMetrics.get(metric.id)?.unit === metric.unit).map(metric => metric.id);
  if (!sharedMetricIds.length) return { compatible: false, reason: "No measured metric with the same definition and unit is shared.", sharedMetricIds: [] };
  return { compatible: true, reason: "Same task, dataset, split, metric definition, and unit.", sharedMetricIds };
}

export function metricsForComparison(record: BenchmarkResult, ids: string[]): BenchmarkMetric[] {
  return ids.map(id => record.metrics.find(metric => metric.id === id)).filter((metric): metric is BenchmarkMetric => Boolean(metric));
}
