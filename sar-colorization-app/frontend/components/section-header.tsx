import { ReactNode } from "react";
export function SectionHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return <header className="mb-10 max-w-3xl"><p className="eyebrow">{eyebrow}</p><h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">{title}</h1><p className="muted mt-4">{children}</p></header>;
}
