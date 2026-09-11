import type { Metadata } from "next";
import { PresentationMode } from "@/components/presentation-mode";

export const metadata: Metadata = {
  title: "Presentation Mode",
  description: "SatQuery AI guided product presentation.",
};

export default function PresentationPage() {
  return <PresentationMode/>;
}
