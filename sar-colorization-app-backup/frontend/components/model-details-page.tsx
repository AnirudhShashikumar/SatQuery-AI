"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, CheckCircle2, CircleAlert, Cpu, Database, Gauge, GitBranch, Layers3, ShieldAlert } from "lucide-react";
import Link from "next/link";
import type { ModelDefinition } from "@/lib/model-catalog";
import { cn } from "@/lib/utils";
import { getAgentHealth, getHealth } from "@/services/api";

function humanize(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());
}

export function ModelDetailsPage({ model }: { model: ModelDefinition }) {
  const backend = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: false });
  const agent = useQuery({ queryKey: ["agent-health"], queryFn: () => getAgentHealth(), retry: false });
  const backendStatus = model.modelHealthKey ? backend.data?.models[model.modelHealthKey] : undefined;
  const specialistStatus = model.specialistHealthKey ? agent.data?.specialists[model.specialistHealthKey] : undefined;
  const available = backendStatus?.available ?? (specialistStatus ? ["ready", "loaded", "available"].includes(specialistStatus.status) : null);
  const status = backendStatus
    ? (backendStatus.available ? "Ready" : "Unavailable")
    : specialistStatus
      ? humanize(specialistStatus.status)
      : backend.isLoading || agent.isLoading
        ? "Checking runtime"
        : "Reported during workflow";
  const checkpoint = backendStatus?.checkpoint || model.checkpoint;
  const runtime = specialistStatus?.device || backend.data?.device || "Selected by backend";

  return <section className="model-detail-page" aria-labelledby="model-detail-title">
    <header className="model-detail-hero">
      <div>
        <p className="eyebrow">Model registry · verified integration</p>
        <h1 id="model-detail-title">{model.name}</h1>
        <p className="model-detail-role">{model.role}</p>
        <p className="model-detail-summary">{model.summary}</p>
      </div>
      <div className={cn("model-runtime-status", available === true && "is-ready", available === false && "is-unavailable")} role="status">
        {available === false ? <CircleAlert size={16}/> : <CheckCircle2 size={16}/>}<span><small>Runtime status</small><strong>{status}</strong></span>
      </div>
    </header>

    <div className="model-detail-grid">
      <ModelFact icon={GitBranch} label="Integration" value={model.integration}/>
      <ModelFact icon={Database} label="Checkpoint / version" value={checkpoint}/>
      <ModelFact icon={Cpu} label="Runtime / device" value={runtime}/>
      <ModelFact icon={Layers3} label="Output" value={model.output}/>
    </div>

    <div className="model-detail-columns">
      <article className="panel model-detail-panel">
        <div className="model-detail-panel-title"><Gauge size={17}/><h2>Supported inputs</h2></div>
        <ul>{model.inputs.map(input => <li key={input}>{input}</li>)}</ul>
      </article>
      <article className="panel model-detail-panel">
        <div className="model-detail-panel-title"><GitBranch size={17}/><h2>Model provenance</h2></div>
        <p>{model.provenance}</p>
        {model.supportingStack && <ul className="mt-4">{model.supportingStack.map(item => <li key={item}>{item}</li>)}</ul>}
      </article>
      <article className="panel model-detail-panel model-detail-limitations">
        <div className="model-detail-panel-title"><ShieldAlert size={17}/><h2>Limitations</h2></div>
        <ul>{model.limitations.map(item => <li key={item}>{item}</li>)}</ul>
      </article>
    </div>

    {available === false && <p className="model-unavailable-note">The runtime reported this optional model as unavailable. Its page remains accessible so checkpoint state and workflow limitations stay visible.</p>}
    <Link href={model.actionHref} className="model-detail-action">{model.actionLabel}<ArrowRight size={16}/></Link>
  </section>;
}

function ModelFact({ icon: Icon, label, value }: { icon: typeof Cpu; label: string; value: string }) {
  return <article><span><Icon size={17}/></span><div><p>{label}</p><strong>{value}</strong></div></article>;
}
