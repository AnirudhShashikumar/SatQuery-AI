import type { Metadata } from "next";
import { ComplianceDashboard } from "@/components/compliance-dashboard";
import { ProductShell } from "@/components/product-shell";
import { PresentationReturnBanner } from "@/components/presentation-page-handoff";

export const metadata: Metadata = { title: "SatQuery Compliance" };

export default function CompliancePage() {
  return <ProductShell><PresentationReturnBanner route="/assistant/compliance"/><ComplianceDashboard/></ProductShell>;
}
