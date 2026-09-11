"use client";

import { motion } from "framer-motion";
import { Suspense } from "react";
import { Sidebar } from "@/components/sidebar";

export function ProductShell({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <main className="product-shell"><Suspense fallback={null}><Sidebar/></Suspense><motion.section initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .38, ease: [0.22, 1, 0.36, 1] }} className={`product-content ${className}`}>{children}</motion.section></main>;
}
