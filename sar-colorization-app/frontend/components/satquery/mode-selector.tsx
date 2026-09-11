"use client";

import { CalendarRange, Images, Layers3 } from "lucide-react";
import type { InputMode } from "@/types/agent";
import { cn } from "@/lib/utils";

const workflows = [
  {
    id: "single" as const,
    title: "Single Image",
    description: "Describe scenes, answer questions, or localize objects.",
    requirement: "One optical, multispectral, or SAR observation",
    example: "Is water visible?",
    icon: Images,
  },
  {
    id: "cross_modal" as const,
    title: "Optical + SAR",
    description: "Combine spectral and structural evidence.",
    requirement: "One optical image and one SAR observation",
    example: "Where do both modalities agree?",
    icon: Layers3,
  },
  {
    id: "bi_temporal" as const,
    title: "Bi-temporal",
    description: "Detect and explain changes between observations.",
    requirement: "Earlier and later observations with dates",
    example: "What changed between these dates?",
    icon: CalendarRange,
  },
] satisfies Array<{ id: InputMode; title: string; description: string; requirement: string; example: string; icon: typeof Images }>;

export function ModeSelector({ value, onChange, disabled = false }: { value: InputMode; onChange: (mode: InputMode) => void; disabled?: boolean }) {
  return <fieldset className="sq-mode-selector" disabled={disabled}>
    <legend className="sr-only">Select analysis workflow</legend>
    {workflows.map(workflow => {
      const Icon = workflow.icon;
      const active = value === workflow.id;
      return <button
        key={workflow.id}
        type="button"
        className={cn("sq-mode-card", active && "sq-mode-card-active")}
        aria-pressed={active}
        onClick={() => onChange(workflow.id)}
      >
        <span className="sq-mode-icon"><Icon size={18}/></span>
        <span className="sq-mode-copy"><strong>{workflow.title}</strong><small>{workflow.description}</small></span>
        <span className="sq-mode-meta"><span>{workflow.requirement}</span><em>{workflow.example}</em></span>
        <span className="sq-mode-state" aria-hidden="true">{active ? "Active" : "Select"}</span>
      </button>;
    })}
  </fieldset>;
}

export const workflowLabel = (mode: InputMode) => workflows.find(workflow => workflow.id === mode)?.title ?? "Single Image";
export const workflowDescription = (mode: InputMode) => workflows.find(workflow => workflow.id === mode)?.description ?? workflows[0].description;
