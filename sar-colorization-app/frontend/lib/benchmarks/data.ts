import suiteSource from "../../../benchmarks/suites/satquery_benchmark_suite.json";
import demosSource from "../../../benchmarks/demos/demo_gallery.json";
import rsvqaTest from "../../../benchmarks/results/rsvqa/rsvqa.specialist-v1.official-test.2026-08-29.json";
import rsvqaValidation from "../../../benchmarks/results/rsvqa/rsvqa.specialist-v1.validation.epoch8.json";
import groundingValidation from "../../../benchmarks/results/grounding/grounding.specialist-v1-1.vrsbench-validation.step600.json";
import groundingSmoke from "../../../benchmarks/results/grounding/grounding.dino-tiny.vrsbench-smoke100.baseline.json";
import changerexOperational from "../../../benchmarks/results/changerex/changerex.r18.hanford-mps-operational.json";
import pix2pixOperational from "../../../benchmarks/results/sar_translation/sar-translation.pix2pix.single-image-operational.json";
import captionUnavailable from "../../../benchmarks/results/captioning/captioning.unavailable.v1.json";
import sarFusionUnavailable from "../../../benchmarks/results/sar_translation/sar_translation_sarfusionformer.unavailable.v1.json";
import crossModalUnavailable from "../../../benchmarks/results/cross_modal/cross_modal.unavailable.v1.json";
import { validateBenchmarkRecord, validateBenchmarkSuite, validateDemoManifest } from "./validation";
import type { BenchmarkResult, LoadedBenchmarkData } from "./types";

const sources: unknown[] = [rsvqaTest, rsvqaValidation, groundingValidation, groundingSmoke, changerexOperational, pix2pixOperational, captionUnavailable, sarFusionUnavailable, crossModalUnavailable];
const statusRank: Record<string, number> = { verified_test: 8, verified_validation: 7, partial_validation: 6, external_reported: 5, smoke_only: 4, operational_only: 3, unavailable: 2, not_applicable: 1 };

export function selectPreferredRun(records: BenchmarkResult[], preferredId?: string): BenchmarkResult | undefined {
  if (preferredId) {
    const explicit = records.find(record => record.benchmark_id === preferredId);
    if (explicit) return explicit;
  }
  return [...records].sort((a, b) => {
    const rank = (statusRank[b.evaluation.status] ?? 0) - (statusRank[a.evaluation.status] ?? 0);
    if (rank) return rank;
    return (b.evaluation.timestamp ?? "").localeCompare(a.evaluation.timestamp ?? "");
  })[0];
}

export function loadBenchmarkData(): LoadedBenchmarkData {
  const warnings: string[] = [];
  const suiteValidation = validateBenchmarkSuite(suiteSource);
  if (!suiteValidation.valid) throw new Error(`Benchmark suite is invalid: ${suiteValidation.errors.join(", ")}`);
  const records = sources.flatMap((source, index) => {
    const validation = validateBenchmarkRecord(source);
    if (!validation.valid) {
      warnings.push(`Benchmark record ${index + 1} rejected: ${validation.errors.join(", ")}`);
      return [];
    }
    return [validation.value];
  });
  const demoValidation = validateDemoManifest(demosSource);
  if (!demoValidation.valid) throw new Error(`Demo manifest is invalid: ${demoValidation.errors.join(", ")}`);
  const historyBySpecialist: Record<string, BenchmarkResult[]> = {};
  for (const record of records) (historyBySpecialist[record.specialist_id] ??= []).push(record);
  for (const history of Object.values(historyBySpecialist)) history.sort((a, b) => (b.evaluation.timestamp ?? "").localeCompare(a.evaluation.timestamp ?? ""));
  const preferred = suiteValidation.value.specialists.flatMap(specialist => {
    const record = selectPreferredRun(historyBySpecialist[specialist] ?? [], suiteValidation.value.preferred_record_by_specialist[specialist]);
    if (!record) warnings.push(`No valid record for ${specialist}`);
    return record ? [record] : [];
  });
  return { suite: suiteValidation.value, records, preferred, historyBySpecialist, warnings, demos: demoValidation.value };
}

export const benchmarkData = loadBenchmarkData();
