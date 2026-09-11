import type { Metadata } from "next";
import { MissionComparisonWorkspace } from "@/components/mission-comparison-workspace";
import { ProductShell } from "@/components/product-shell";

export const metadata: Metadata = { title: "Mission Comparison" };

export default function MissionComparisonPage() {
  return <ProductShell><MissionComparisonWorkspace/></ProductShell>;
}
