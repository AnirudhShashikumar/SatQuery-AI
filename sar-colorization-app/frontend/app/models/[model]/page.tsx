import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { ModelDetailsPage } from "@/components/model-details-page";
import { ProductShell } from "@/components/product-shell";
import { isModelSlug, modelCatalog } from "@/lib/model-catalog";

export function generateStaticParams() {
  return Object.keys(modelCatalog).map(model => ({ model }));
}

export async function generateMetadata({ params }: { params: Promise<{ model: string }> }): Promise<Metadata> {
  const { model } = await params;
  if (!isModelSlug(model)) return { title: "Model not found" };
  return { title: modelCatalog[model].name, description: modelCatalog[model].role };
}

export default async function ModelPage({ params }: { params: Promise<{ model: string }> }) {
  const { model } = await params;
  if (!isModelSlug(model)) notFound();
  return <ProductShell><ModelDetailsPage model={modelCatalog[model]}/></ProductShell>;
}
