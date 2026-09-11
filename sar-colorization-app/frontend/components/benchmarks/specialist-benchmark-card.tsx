import { Activity, CalendarDays, Cpu, Database, FileCheck2, Fingerprint, Gauge, History, MemoryStick, Network, ShieldAlert } from "lucide-react";
import { BenchmarkStatusBadge } from "./benchmark-status-badge";
import { MetricGrid } from "./metric-grid";
import { MissingBenchmarkState } from "./missing-benchmark-state";
import { formatDatasetSplit, formatMetricValue, formatTimestamp, shortFingerprint, statusMeta, type BenchmarkResult } from "@/lib/benchmarks";

function Fact({ icon: Icon, label, value }: { icon: typeof Database; label: string; value: string }) {
  return <div className="benchmark-fact"><Icon size={14}/><span><small>{label}</small><strong>{value}</strong></span></div>;
}

function BreakdownChart({ record }: { record: BenchmarkResult }) {
  const rows = (record.breakdowns ?? []).filter(row => row.metrics.some(metric => metric.id === "accuracy" || metric.id === "accuracy_at_050")).slice(0, record.specialist_id === "grounding" ? 12 : 8);
  if (!rows.length) return null;
  const label = record.specialist_id === "grounding" ? "Per-class accuracy @ IoU 0.50" : "Accuracy by question family";
  return <div className="benchmark-breakdown"><h4>{label}</h4><div className="benchmark-breakdown-bars" role="img" aria-label={`${label}. A table alternative follows.`}>{rows.map(row => { const value = row.metrics.find(metric => metric.id === "accuracy" || metric.id === "accuracy_at_050")?.value ?? 0; return <div key={row.label}><span>{row.label.replaceAll("_", " ")}</span><i><b style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}/></i><strong>{(value * 100).toFixed(1)}%</strong></div>; })}</div><details><summary>Accessible data table</summary><table><thead><tr><th>Group</th><th>Samples</th><th>Accuracy</th></tr></thead><tbody>{rows.map(row => { const value = row.metrics.find(metric => metric.id === "accuracy" || metric.id === "accuracy_at_050"); return <tr key={row.label}><th>{row.label.replaceAll("_", " ")}</th><td>{row.sample_count ?? "—"}</td><td>{value ? formatMetricValue(value.value, value.unit) : "Unavailable"}</td></tr>; })}</tbody></table></details></div>;
}

export function SpecialistBenchmarkCard({ record, history = [] }: { record: BenchmarkResult; history?: BenchmarkResult[] }) {
  const status = statusMeta[record.evaluation.status];
  const environmentLabel = [record.evaluation.environment.device, record.evaluation.environment.hardware].filter(Boolean).join(" · ") || "Not recorded";
  return <article className="specialist-benchmark-card" id={`benchmark-${record.specialist_id}`} aria-labelledby={`benchmark-${record.specialist_id}-title`}><header><div><div className="benchmark-card-kicker"><span>{record.task}</span><BenchmarkStatusBadge status={record.evaluation.status}/></div><h3 id={`benchmark-${record.specialist_id}-title`}>{record.display_name}</h3><p>{record.model.architecture}</p></div><span className="benchmark-card-index" aria-hidden="true">{String(record.display_name).slice(0, 2).toUpperCase()}</span></header>
    {record.evaluation.status === "unavailable" ? <MissingBenchmarkState note={record.notes[0]}/> : <>
      <MetricGrid metrics={record.metrics.slice(0, 7)} statusLabel={status.label}/>
      <div className="benchmark-fact-grid"><Fact icon={Database} label="Evaluation data" value={formatDatasetSplit(record.evaluation.dataset, record.evaluation.split)}/><Fact icon={Activity} label="Sample count" value={record.evaluation.sample_count?.toLocaleString() ?? "Not recorded"}/><Fact icon={Cpu} label="Device / hardware" value={environmentLabel}/><Fact icon={CalendarDays} label="Measured" value={formatTimestamp(record.evaluation.timestamp)}/></div>
      <BreakdownChart record={record}/>
      <details className="benchmark-details"><summary>Protocol, provenance, and limitations</summary><div className="benchmark-detail-grid"><section><h4><Network size={14}/>Protocol</h4><p>{record.evaluation.protocol}</p><dl><div><dt>Model</dt><dd>{record.model.name} · {record.model.version}</dd></div><div><dt>Training / adaptation</dt><dd>{[record.model.training_dataset, record.model.adaptation].filter(Boolean).join(" · ") || "Not recorded"}</dd></div><div><dt>Input context</dt><dd>{record.evaluation.environment.input_size ?? "Not recorded"}</dd></div></dl></section><section><h4><Fingerprint size={14}/>Checkpoint provenance</h4><dl><div><dt>Checkpoint</dt><dd className="benchmark-wrap">{record.model.checkpoint ?? "Not recorded"}</dd></div><div><dt>SHA-256</dt><dd className="benchmark-wrap" title={record.model.checkpoint_sha256 ?? undefined}>{shortFingerprint(record.model.checkpoint_sha256)}</dd></div><div><dt>Result origin</dt><dd>{record.provenance.result_origin.replaceAll("_", " ")}</dd></div><div><dt>Source artifacts</dt><dd>{record.provenance.source_artifacts.map(source => source.split("/").at(-1)).join(" · ")}</dd></div></dl></section><section><h4><ShieldAlert size={14}/>Limitations</h4><ul>{record.limitations.slice(0, 5).map(item => <li key={item}>{item}</li>)}</ul></section></div></details>
    </>}
    <footer><span><Gauge size={13}/>{record.performance.mean_latency_ms === null ? "Latency unavailable" : `${record.performance.mean_latency_ms.toFixed(2)} ms mean`}</span><span><MemoryStick size={13}/>{record.performance.peak_memory_mb === null ? "Memory unavailable" : `${record.performance.peak_memory_mb.toFixed(0)} MB peak`}</span><span><FileCheck2 size={13}/>{record.provenance.verified ? "Source artifact verified" : "No measured artifact"}</span>{history.length > 1 && <span><History size={13}/>{history.length} historical runs</span>}</footer>
  </article>;
}
