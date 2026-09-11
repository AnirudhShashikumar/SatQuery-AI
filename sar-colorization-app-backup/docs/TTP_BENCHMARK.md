# TTP benchmark protocol

No official-checkpoint benchmark has been executed in this Apple Silicon workspace because the pinned service requires Python 3.10, CUDA, and the approximately 1.2 GiB checkpoint. Runtime, GPU memory, IoU, precision, recall, F1, false-positive rate, false-negative rate, and changed-area error are therefore **not measured here**.

On the persistent CUDA host, run `scripts/benchmark_ttp.py` with a manifest covering the five bundled official samples, a LEVIR-CD test subset when licensed/available, the approved GeoVision demo pair, unchanged pairs, seasonal variation, misaligned pairs, non-building changes, and unsupported SAR pairs. Misaligned and SAR cases are eligibility/fallback checks, not model-accuracy samples.

The harness writes `summary.json`, `per_sample.csv`, `per_category.csv`, `failure_cases.csv`, `benchmark_report.md`, and `visual_contact_sheet.png`. It reports TTP metrics only where a label exists and keeps deterministic and learned metrics separate. It never hides failures or selects a demonstration pair automatically.

Demo approval criteria:

1. The warmed TTP mask is visually strong and supported by label or defensible visual evidence.
2. Runtime and GPU memory are measured on the presentation host.
3. The deterministic fallback completes for the same pair.
4. The report package contains all labelled evidence.
5. Weak and failed samples remain in the benchmark outputs.
