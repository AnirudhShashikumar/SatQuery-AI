"use client";

import { Download, FileJson2, FileSpreadsheet, FileText } from "lucide-react";
import { jsPDF } from "jspdf";
import { benchmarkPdfLines, benchmarkRecordsToCsv, buildBenchmarkJsonExport, type BenchmarkResult, type BenchmarkSuite } from "@/lib/benchmarks";

function download(name: string, contents: BlobPart, type: string) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = name; anchor.click();
  URL.revokeObjectURL(url);
}

export function BenchmarkExportPanel({ suite, records }: { suite: BenchmarkSuite; records: BenchmarkResult[] }) {
  const json = () => download("satquery-benchmark-suite.json", JSON.stringify(buildBenchmarkJsonExport(suite, records), null, 2), "application/json");
  const csv = () => download("satquery-benchmark-suite.csv", benchmarkRecordsToCsv(records), "text/csv;charset=utf-8");
  const pdf = () => {
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    let y = 46;
    for (const line of benchmarkPdfLines(suite, records)) {
      const wrapped = doc.splitTextToSize(line || " ", 500) as string[];
      if (y + wrapped.length * 14 > 790) { doc.addPage(); y = 46; }
      doc.setFont("helvetica", line.startsWith("SatQuery") ? "bold" : "normal");
      doc.setFontSize(line.startsWith("SatQuery") ? 16 : 9);
      doc.text(wrapped, 48, y); y += Math.max(12, wrapped.length * 13);
    }
    doc.save("satquery-benchmark-suite.pdf");
  };
  return <section className="benchmark-export-panel" aria-labelledby="benchmark-export-title"><div><p className="eyebrow">Reproducible exports</p><h2 id="benchmark-export-title">Export the normalized evidence</h2><p>Files are generated from validated benchmark records—not scraped from the rendered dashboard.</p></div><div><button onClick={json}><FileJson2 size={16}/>JSON<span>Full provenance</span></button><button onClick={csv}><FileSpreadsheet size={16}/>CSV<span>One metric per row</span></button><button onClick={pdf}><FileText size={16}/>PDF<span>Review summary</span></button></div><small><Download size={13}/>Exports retain benchmark status, dataset, split, timestamp, hardware context, limitations, and source artifacts.</small></section>;
}
