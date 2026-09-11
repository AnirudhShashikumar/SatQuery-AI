import { ImageIcon, ShieldCheck } from "lucide-react";
import { benchmarkData } from "@/lib/benchmarks";
import { DemoGallery } from "@/components/benchmarks/demo-gallery";

export function DemoGalleryView() {
  return <section className="demo-gallery-page"><header className="benchmark-header"><div><p className="eyebrow">Curated evidence gallery</p><h1>Demo Gallery</h1><p>Attributed remote-sensing cases with cautious expected behavior and guided access to the real SatQuery workspace.</p></div><span><ShieldCheck size={15}/>{benchmarkData.demos.cases.length} verified assets</span></header><div className="demo-gallery-disclosure"><ImageIcon size={19}/><div><strong>No prerecorded model answers</strong><p>Gallery assets and annotations provide context only. Guided launch uses production routing and can return a different or null model result.</p></div></div><DemoGallery cases={benchmarkData.demos.cases}/></section>;
}
