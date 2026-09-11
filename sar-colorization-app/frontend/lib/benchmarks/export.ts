import type { BenchmarkResult, BenchmarkSuite } from "./types";
import { formatMetricValue, statusMeta } from "./format";

export type BenchmarkExport = {
  schema_version: string;
  suite: Pick<BenchmarkSuite, "suite_version" | "name" | "generated_at" | "source_commit" | "known_gaps" | "reproducibility_notes">;
  exported_at: string;
  records: BenchmarkResult[];
};

export function buildBenchmarkJsonExport(suite: BenchmarkSuite, records: BenchmarkResult[], exportedAt = new Date().toISOString()): BenchmarkExport {
  return { schema_version: suite.schema_version, suite: { suite_version: suite.suite_version, name: suite.name, generated_at: suite.generated_at, source_commit: suite.source_commit, known_gaps: suite.known_gaps, reproducibility_notes: suite.reproducibility_notes }, exported_at: exportedAt, records };
}

const csvCell = (value: string | number | null | boolean) => `"${String(value ?? "").replaceAll('"', '""')}"`;

export function benchmarkRecordsToCsv(records: BenchmarkResult[]): string {
  const headers = ["specialist_id", "model_name", "model_version", "task", "benchmark_status", "dataset", "split", "sample_count", "metric_id", "metric_label", "metric_value", "metric_unit", "mean_latency_ms", "p95_latency_ms", "peak_memory_mb", "device", "hardware", "checkpoint", "timestamp", "result_origin"];
  const rows = records.flatMap(record => {
    const metrics = record.metrics.length ? record.metrics : [null];
    return metrics.map(metric => [record.specialist_id, record.model.name, record.model.version, record.task, record.evaluation.status, record.evaluation.dataset, record.evaluation.split, record.evaluation.sample_count, metric?.id ?? null, metric?.label ?? null, metric?.value ?? null, metric?.unit ?? null, record.performance.mean_latency_ms, record.performance.p95_latency_ms, record.performance.peak_memory_mb, record.evaluation.environment.device, record.evaluation.environment.hardware, record.model.checkpoint, record.evaluation.timestamp, record.provenance.result_origin]);
  });
  return [headers, ...rows].map(row => row.map(csvCell).join(",")).join("\n") + "\n";
}

export function benchmarkPdfLines(suite: BenchmarkSuite, records: BenchmarkResult[], generatedAt = new Date().toISOString()): string[] {
  const lines = ["SatQuery AI — Benchmark Summary", `Suite ${suite.suite_version} · generated ${generatedAt}`, "Validation, test, smoke, and operational statuses are not interchangeable."];
  for (const record of records) {
    lines.push("", `${record.display_name} — ${statusMeta[record.evaluation.status].label}`, `${record.evaluation.dataset ?? "Dataset unavailable"} · ${record.evaluation.split ?? "Split unavailable"} · n=${record.evaluation.sample_count ?? "unavailable"}`);
    for (const metric of record.metrics.filter(item => item.primary).slice(0, 4)) lines.push(`${metric.label}: ${formatMetricValue(metric.value, metric.unit)}`);
    if (!record.metrics.length) lines.push("Quality metrics: unavailable");
    if (record.performance.mean_latency_ms !== null) lines.push(`Mean latency: ${record.performance.mean_latency_ms.toFixed(2)} ms on ${record.evaluation.environment.hardware ?? record.evaluation.environment.device ?? "undisclosed hardware"}`);
    lines.push(`Provenance: ${record.provenance.result_origin}; ${record.provenance.source_artifacts.join(", ")}`);
    if (record.limitations[0]) lines.push(`Limitation: ${record.limitations[0]}`);
  }
  lines.push("", "Footnote: verified validation is not test performance. Smoke and operational records do not establish model accuracy.");
  return lines;
}
