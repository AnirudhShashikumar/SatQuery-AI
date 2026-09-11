"use client";

import { motion } from "framer-motion";
import { ArrowRight, Bot, ChevronRight, CircleDot, History, Images, Layers3, MessageSquareText, Paperclip, Radar, Satellite, ScanSearch, Sparkles, Waves } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { readRecentConversations, rememberConversation, type RecentConversation } from "@/lib/recent-conversations";

const prompts = ["Describe this satellite image", "What changed between these two dates?", "Compare the optical and SAR observations", "Count the buildings in this scene", "Highlight every visible water body", "Is this area urban or rural?"] as const;
const capabilities = [
  { icon: Images, title: "Single image reasoning", copy: "Ask direct questions about land cover, infrastructure, and spatial context." },
  { icon: Layers3, title: "Optical + SAR fusion", copy: "Connect spectral appearance with structure that remains visible through cloud and darkness." },
  { icon: ScanSearch, title: "Grounded evidence", copy: "Move from a prose answer to traceable regions, confidence, and specialist evidence." },
] as const;

export function LandingPage() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [recent, setRecent] = useState<RecentConversation[]>([]);
  useEffect(() => setRecent(readRecentConversations(window.localStorage)), []);
  function ask(prompt: string) { const clean = prompt.trim(); if (clean) rememberConversation(window.localStorage, clean); router.push(clean ? `/assistant?prompt=${encodeURIComponent(clean)}` : "/assistant"); }
  function submit(event: FormEvent) { event.preventDefault(); ask(query); }

  return <main className="landing-page">
    <div className="landing-aurora" aria-hidden="true"/>
    <nav className="landing-nav" aria-label="Primary navigation">
      <Link href="/" className="landing-brand" aria-label="SatQuery AI home"><span className="sat-brand-mark" aria-hidden="true"><span/><span/><span/></span><span className="sat-brand-name"><strong>SatQuery</strong><span>AI</span></span></Link>
      <div className="landing-nav-links"><Link href="/benchmark">Benchmarks</Link><Link href="/about">About</Link><Link href="/assistant" className="landing-launch">Launch Assistant <ArrowRight size={15}/></Link></div>
    </nav>
    <section className="landing-hero">
      <motion.div className="landing-copy" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .65, ease: [0.22, 1, 0.36, 1] }}>
        <span className="landing-status"><CircleDot size={13}/> Multimodal geospatial intelligence</span>
        <h1>Ask Earth <span>Anything.</span></h1>
        <p className="landing-subtitle">Interactive Vision-Language Assistant for Multimodal Remote Sensing Intelligence</p>
        <p className="landing-description">Turn satellite imagery into grounded answers, comparisons, change evidence, and decision-ready reports through one spatial reasoning interface.</p>
        <form className="landing-prompt" onSubmit={submit}>
          <div className="landing-prompt-heading"><Bot size={16}/><span>Ask SatQuery</span><small>Agent online</small></div>
          <div className="landing-input-row"><textarea value={query} onChange={event => setQuery(event.target.value)} rows={2} aria-label="Ask a remote sensing question" placeholder="Describe this satellite image..."/><button type="button" className="landing-attach" onClick={() => router.push("/assistant?attach=1")} aria-label="Attach imagery in the assistant"><Paperclip size={17}/></button><button type="submit" aria-label="Open this question in the assistant"><ArrowRight size={19}/></button></div>
        </form>
        <div className="landing-suggestions" aria-label="Suggested questions">{prompts.map(prompt => <button key={prompt} type="button" onClick={() => ask(prompt)}>{prompt}<ChevronRight size={14}/></button>)}</div>
      </motion.div>
      <motion.div className="landing-orbit" aria-hidden="true" initial={{ opacity: 0, scale: .92 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: .8, delay: .14 }}>
        <div className="orbit-ring orbit-ring-one"/><div className="orbit-ring orbit-ring-two"/><div className="orbit-core"><Satellite size={39}/><span>LIVE</span></div>
        <div className="orbit-card orbit-card-one"><Radar size={17}/><span><strong>SAR</strong><small>Structural signal</small></span></div>
        <div className="orbit-card orbit-card-two"><Waves size={17}/><span><strong>Optical</strong><small>Spectral context</small></span></div>
        <div className="orbit-card orbit-card-three"><Sparkles size={17}/><span><strong>SatQuery</strong><small>Grounded synthesis</small></span></div><div className="orbit-pulse"/>
      </motion.div>
    </section>
    <section className="landing-recent" aria-labelledby="recent-conversations-title">
      <header><div><span><History size={15}/></span><div><p>Continue exploring</p><h2 id="recent-conversations-title">Recent conversations</h2></div></div><Link href="/assistant?view=history">View history <ArrowRight size={14}/></Link></header>
      {recent.length > 0 ? <div className="landing-recent-grid">{recent.slice(0, 3).map(item => <button key={item.id} type="button" onClick={() => ask(item.prompt)}><span><MessageSquareText size={16}/></span><div><strong>{item.prompt}</strong><small>{item.mode ? item.mode.replaceAll("_", " ") : "Earth intelligence"} · {new Date(item.createdAt).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</small></div><ChevronRight size={15}/></button>)}</div> : <div className="landing-recent-empty"><span><MessageSquareText size={18}/></span><div><strong>Your Earth questions will live here.</strong><p>Start a conversation to quickly return to imagery, evidence, and results.</p></div><button type="button" onClick={() => ask("")}>Start a conversation <ArrowRight size={14}/></button></div>}
    </section>
    <section className="landing-capabilities" aria-label="Core capabilities">{capabilities.map((capability, index) => { const Icon = capability.icon; return <motion.article key={capability.title} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: .3 + index * .08 }}><span><Icon size={19}/></span><div><h2>{capability.title}</h2><p>{capability.copy}</p></div></motion.article>; })}</section>
  </main>;
}
