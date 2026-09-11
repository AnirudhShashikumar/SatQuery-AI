export const benchmarkStatuses = [
  "verified_test", "verified_validation", "partial_validation", "smoke_only",
  "operational_only", "external_reported", "unavailable", "not_applicable",
] as const;

export type BenchmarkStatus = typeof benchmarkStatuses[number];
export type MetricUnit = "ratio" | "percentage" | "milliseconds" | "seconds" | "megabytes" | "count" | "samples_per_second" | "db" | "score";

export type BenchmarkMetric = {
  id: string;
  label: string;
  value: number | null;
  unit: MetricUnit;
  higher_is_better: boolean | null;
  description: string;
  primary: boolean;
};

export type BenchmarkEnvironment = {
  device: string | null;
  hardware: string | null;
  os: string | null;
  python: string | null;
  torch: string | null;
  input_size: string | null;
};

export type BenchmarkResult = {
  schema_version: "1.0.0";
  benchmark_id: string;
  specialist_id: string;
  display_name: string;
  task: string;
  model: {
    name: string;
    version: string;
    architecture: string;
    checkpoint: string | null;
    checkpoint_sha256: string | null;
    training_dataset: string | null;
    adaptation: string | null;
    license: string | null;
    source: string | null;
  };
  evaluation: {
    status: BenchmarkStatus;
    dataset: string | null;
    split: string | null;
    sample_count: number | null;
    protocol: string;
    timestamp: string | null;
    environment: BenchmarkEnvironment;
  };
  metrics: BenchmarkMetric[];
  performance: {
    mean_latency_ms: number | null;
    median_latency_ms: number | null;
    p95_latency_ms: number | null;
    throughput_samples_per_second: number | null;
    peak_memory_mb: number | null;
    model_load_ms: number | null;
    warm_reuse: boolean | null;
  };
  breakdowns?: Array<{
    dimension: string;
    label: string;
    sample_count: number | null;
    metrics: BenchmarkMetric[];
  }>;
  artifacts: {
    report: string | null;
    predictions: string | null;
    per_class_metrics: string | null;
    confusion_matrix: string | null;
    visual_examples: string[];
  };
  limitations: string[];
  notes: string[];
  provenance: {
    generated_by: string;
    source_artifacts: string[];
    verified: boolean;
    result_origin: "satquery_reproduced" | "external_reported" | "unmeasured";
  };
};

export type BenchmarkSuite = {
  schema_version: "1.0.0";
  suite_version: string;
  name: string;
  generated_at: string;
  source_commit: string;
  specialists: string[];
  benchmark_record_ids: string[];
  preferred_record_by_specialist: Record<string, string>;
  coverage: Record<Exclude<BenchmarkStatus, "not_applicable">, number>;
  known_gaps: string[];
  reproducibility_notes: string[];
};

export type DemoResultClassification = "benchmark_ground_truth" | "curated_expected_behavior" | "illustrative_only";
export type DemoCase = {
  demo_id: string;
  title: string;
  workflow: "single_image" | "grounding" | "bi_temporal" | "cross_modal" | "sar_translation";
  category: string;
  primary_image: string;
  secondary_image: string | null;
  modality: string;
  dates: string[];
  question: string;
  expected_response_type: string;
  expected_visual_output_type: string;
  expected_result_classification: DemoResultClassification;
  expected_behavior: string;
  purpose: string;
  difficulty: "introductory" | "moderate" | "challenging";
  limitations: string[];
  source: string;
  attribution: string;
  license: string;
  required_specialists: string[];
  readiness: "ready" | "guided_only" | "unavailable";
  launch_path: string;
};
export type DemoManifest = { schema_version: "1.0.0"; gallery_version: string; generated_at: string; cases: DemoCase[] };

export type LoadedBenchmarkData = {
  suite: BenchmarkSuite;
  records: BenchmarkResult[];
  preferred: BenchmarkResult[];
  historyBySpecialist: Record<string, BenchmarkResult[]>;
  warnings: string[];
  demos: DemoManifest;
};
