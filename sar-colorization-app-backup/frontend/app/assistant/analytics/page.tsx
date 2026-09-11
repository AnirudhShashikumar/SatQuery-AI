import type { Metadata } from "next";
import { ResearchAnalyticsDashboard } from "@/components/research-analytics-dashboard";
import { ProductShell } from "@/components/product-shell";
import { PresentationReturnBanner } from "@/components/presentation-page-handoff";

export const metadata: Metadata = { title: "SatQuery Research Analytics" };

export default function AnalyticsPage() {
  return <ProductShell><PresentationReturnBanner route="/assistant/analytics"/><ResearchAnalyticsDashboard/></ProductShell>;
}
