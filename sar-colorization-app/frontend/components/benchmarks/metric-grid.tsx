import { Info } from "lucide-react";
import { formatMetricValue, type BenchmarkMetric } from "@/lib/benchmarks";

export function MetricGrid({ metrics, statusLabel }: { metrics: BenchmarkMetric[]; statusLabel: string }) {
  if (!metrics.length) return <p className="benchmark-metric-empty">No quality metric is available for this record.</p>;
  return <dl className="benchmark-metric-grid" aria-label={`Metrics — ${statusLabel}`}>{metrics.map(metric => <div key={metric.id} className={metric.primary ? "is-primary" : ""}><dt><span>{metric.label}</span><span className="benchmark-info" title={metric.description} aria-label={metric.description}><Info size={12}/></span></dt><dd>{formatMetricValue(metric.value, metric.unit)}</dd><small>{metric.primary ? "Primary metric" : metric.higher_is_better === null ? "Descriptive" : metric.higher_is_better ? "Higher is better" : "Lower is better"}</small></div>)}</dl>;
}
