import { AlertTriangle, CheckCircle2, CircleSlash2, Clock3, ExternalLink, FlaskConical } from "lucide-react";
import { statusMeta, type BenchmarkStatus } from "@/lib/benchmarks";

const icons = { verified_test: CheckCircle2, verified_validation: CheckCircle2, partial_validation: AlertTriangle, smoke_only: FlaskConical, operational_only: Clock3, external_reported: ExternalLink, unavailable: CircleSlash2, not_applicable: CircleSlash2 };

export function BenchmarkStatusBadge({ status, compact = false }: { status: BenchmarkStatus; compact?: boolean }) {
  const meta = statusMeta[status];
  const Icon = icons[status];
  return <span className={`benchmark-status benchmark-status-${meta.tone}`} title={meta.description} aria-label={`${meta.label}. ${meta.description}`}><Icon size={13}/>{compact ? meta.short : meta.label}</span>;
}
