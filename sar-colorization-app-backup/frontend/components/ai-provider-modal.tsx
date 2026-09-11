"use client";

import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, Eye, EyeOff, KeyRound, LoaderCircle, ShieldCheck, Sparkles, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { ProviderId } from "@/types/api";
import { getProviderModels, saveProviderSettings } from "@/services/api";

type Props = { open: boolean; onClose: () => void; onSaved: () => void; notice?: string };

const providers: { id: ProviderId; label: string; note: string }[] = [
  { id: "gemini", label: "Google Gemini", note: "Available now" },
  { id: "openai", label: "OpenAI", note: "Existing configuration only" },
  { id: "anthropic", label: "Anthropic Claude", note: "Future support" },
];

export function AIProviderModal({ open, onClose, onSaved, notice }: Props) {
  const [provider, setProvider] = useState<ProviderId>("gemini");
  const models = useQuery({ queryKey: ["gemini-models"], queryFn: getProviderModels, enabled: open });
  const [model, setModel] = useState("");
  useEffect(() => {
    if (!model && models.data?.default_model) setModel(models.data.default_model);
  }, [model, models.data?.default_model]);
  const [privacyAcknowledged, setPrivacyAcknowledged] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [visible, setVisible] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const valid = provider === "gemini" && apiKey.trim().length > 0 && !/[\r\n]/.test(apiKey) && !/^(your|paste|example|api[ _-]?key)/i.test(apiKey.trim()) && Boolean(model) && privacyAcknowledged;
  const save = async () => {
    if (!valid) { setError("API key cannot be empty."); return; }
    setSaving(true); setError("");
    try { await saveProviderSettings(provider, apiKey.trim(), model, privacyAcknowledged); setApiKey(""); onSaved(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not save configuration."); }
    finally { setSaving(false); }
  };
  return <AnimatePresence>{open && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[80] grid place-items-center overflow-y-auto bg-black/70 p-4 backdrop-blur-sm"><motion.section initial={{ opacity: 0, y: 18, scale: .98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 12, scale: .98 }} transition={{ duration: .2 }} role="dialog" aria-modal="true" aria-labelledby="provider-modal-title" className="w-full max-w-lg rounded-3xl border border-sky-300/20 bg-zinc-950/95 p-5 shadow-2xl shadow-sky-500/10 sm:p-7"><div className="flex items-start justify-between gap-5"><div><div className="mb-3 grid h-11 w-11 place-items-center rounded-2xl bg-sky-400/15 text-sky-200"><Sparkles size={20}/></div><h2 id="provider-modal-title" className="text-xl font-semibold">Connect Google Gemini</h2><p className="mt-2 text-sm leading-6 text-zinc-400">Connect your Gemini API key to enable optional AI-assisted interpretation of SatQuery AI reconstruction outputs.</p></div><button onClick={onClose} className="rounded-xl p-2 text-zinc-400 hover:bg-white/10 hover:text-white" aria-label="Close configuration"><X size={18}/></button></div>{notice && <p className="mt-4 rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-3 text-sm text-amber-100">{notice}</p>}<div className="mt-6"><label className="text-sm font-medium text-zinc-200">Provider</label><div className="mt-2 grid gap-2">{providers.map(item => <button key={item.id} type="button" disabled={item.id !== "gemini"} onClick={() => item.id === "gemini" && setProvider(item.id)} className={`flex items-center justify-between rounded-xl border p-3 text-left transition ${provider === item.id ? "border-sky-300/50 bg-sky-400/[.10]" : "border-white/[.08] bg-white/[.025] hover:border-white/20"}`}><span className="text-sm text-zinc-100">{item.label}</span><span className={`text-xs ${item.id === "gemini" ? "text-emerald-300" : "text-zinc-500"}`}>{item.note}</span></button>)}</div></div><div className="mt-5"><label htmlFor="provider-model" className="text-sm font-medium text-zinc-200">Model</label><select id="provider-model" value={model} onChange={event => { setModel(event.target.value); setError(""); }} className="mt-2 w-full rounded-xl border border-white/[.12] bg-white/[.03] px-3 py-3 text-sm text-zinc-100 outline-none focus:border-sky-300/60"><option value="">{models.isLoading ? "Loading supported models…" : "Select a Gemini vision model"}</option>{models.data?.models.filter(item => item.supports_images).map(item => <option key={item.id} value={item.id}>{item.label}{item.recommended ? " · Recommended" : ""}</option>)}</select></div><div className="mt-5"><label htmlFor="provider-api-key" className="text-sm font-medium text-zinc-200">API Key</label><div className="mt-2 flex overflow-hidden rounded-xl border border-white/[.12] bg-white/[.03] focus-within:border-sky-300/60"><KeyRound className="m-3 shrink-0 text-zinc-500" size={18}/><input id="provider-api-key" value={apiKey} onChange={event => { setApiKey(event.target.value); setError(""); }} type={visible ? "text" : "password"} placeholder="Paste your Gemini API key" className="min-w-0 flex-1 bg-transparent py-3 text-sm outline-none placeholder:text-zinc-600" autoComplete="off"/><button onClick={() => setVisible(value => !value)} className="px-3 text-zinc-400 hover:text-white" aria-label={visible ? "Hide API key" : "Show API key"}>{visible ? <EyeOff size={18}/> : <Eye size={18}/>}</button></div>{error && <p className="mt-2 text-sm text-rose-300">{error}</p>}</div><label className="mt-4 flex cursor-pointer items-start gap-3 rounded-xl border border-white/[.08] bg-white/[.025] p-3 text-xs leading-5 text-zinc-300"><input type="checkbox" checked={privacyAcknowledged} onChange={event => setPrivacyAcknowledged(event.target.checked)} className="mt-0.5 accent-sky-300"/><span>I understand that selected images may be sent to Google Gemini when I request AI Analysis.</span></label><div className="mt-4 flex items-start gap-2 rounded-xl border border-sky-400/15 bg-sky-400/[.06] p-3 text-xs leading-5 text-sky-100/80"><ShieldCheck className="mt-0.5 shrink-0 text-sky-300" size={16}/>Your API key is sent only to the SatQuery AI backend and used to call Google Gemini for image analysis. It is never included in model outputs, reports, logs, or frontend source code.</div><div className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-sm"><a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer" className="text-sky-300 hover:text-sky-200">Get OpenAI API Key</a><a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer" className="text-sky-300 hover:text-sky-200">Get Gemini API Key</a></div><div className="mt-7 flex justify-end gap-3"><button onClick={onClose} disabled={saving} className="rounded-xl px-4 py-2.5 text-sm text-zinc-400 hover:bg-white/[.06] hover:text-white">Cancel</button><button onClick={save} disabled={!valid || saving} className="inline-flex items-center gap-2 rounded-xl bg-sky-300 px-4 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-sky-200 disabled:cursor-not-allowed disabled:opacity-40">{saving ? <LoaderCircle className="animate-spin" size={16}/> : <CheckCircle2 size={16}/>}Save and Test</button></div></motion.section></motion.div>}</AnimatePresence>;
}
