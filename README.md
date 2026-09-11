<div align="center">

# 🛰️ SatQuery AI

### Ask Earth Anything. Get Evidence, Not Just Answers.

**An evidence-first, agentic vision-language platform for multimodal remote-sensing analysis**

Satellite imagery → Natural-language queries → Specialist AI models → Visual evidence → Auditable answers

<br />

**Smart India Hackathon 2026 · ISRO Problem Statement 26167**

[Overview](#-overview) •
[Features](#-key-capabilities) •
[Architecture](#-system-architecture) •
[Models](#-ai-specialists) •
[Benchmarks](#-verified-benchmarks) •
[Setup](#-getting-started) •
[Scientific Guardrails](#-scientific-guardrails)

</div>

---

## 🌍 Overview

Remote-sensing analysis usually requires specialist knowledge, multiple tools, sensor-specific preprocessing, and significant manual interpretation.

**SatQuery AI** brings these workflows into a single natural-language interface.

Instead of treating satellite imagery as ordinary pictures, SatQuery validates the observation, identifies its modality, understands the user's question, selects an appropriate remote-sensing specialist, executes the analysis, and returns not only an answer — but also the **evidence, confidence, provenance, limitations, and execution trace** behind it.

> **SatQuery AI is not just a chatbot looking at satellite images.**
>
> It is an agentic orchestration layer designed specifically for Earth-observation workflows.

---

## ✨ Key Capabilities

### 🖼️ Single Image Analysis

Analyze an individual **optical, multispectral, or SAR observation**.

Supports workflows including:

- Remote-sensing visual question answering
- Scene understanding
- Land-cover and object interpretation
- Text-guided object grounding
- Spatial evidence visualization
- SAR-assisted interpretation
- Remote-sensing scene priors

Example:

> **“Describe the land-cover and major objects visible in this image.”**

---

### 🛰️ Optical + SAR Analysis

Analyze spatially corresponding **optical and Synthetic Aperture Radar observations together**.

SatQuery preserves the two sensor modalities independently before combining their evidence.

This enables:

- Optical + SAR evidence fusion
- Built-up-region analysis
- Water-region analysis
- Cross-modal agreement detection
- Complementary sensor interpretation
- Sensor-aware limitations
- Qualitative multimodal analysis when strict quantitative registration cannot be established

Example:

> **“Use the optical and SAR images together to identify built-up and water-covered regions.”**

---

### 🕒 Bi-temporal Change Analysis

Compare observations of the same area captured at different times.

Supports:

- Learned change detection
- Change localization
- Change descriptions
- Change-oriented VQA
- Spatial change evidence
- Semantic interpretation of detected changes
- Deterministic supporting evidence

Example:

> **“What changed between these two dates, and where did the change occur?”**

---

### 🎯 Text-Guided Grounding

Find regions in remote-sensing imagery using natural language.

Example:

> **“Highlight the water body referred to in the query.”**

SatQuery returns localized visual evidence rather than only a textual response.

---

## 🧠 Agentic Remote-Sensing Intelligence

SatQuery dynamically determines how a query should be solved.

```text
                    ┌─────────────────────────────┐
                    │ Natural-Language EO Query   │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │ Scientific Input Validator  │
                    │                             │
                    │ Modality • Format • CRS     │
                    │ Metadata • Grid • Bands     │
                    │ Registration • Compatibility│
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       SatQuery Agent        │
                    │                             │
                    │ Intent → Route → Configure  │
                    └──────────────┬──────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
               ▼                   ▼                   ▼
        ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
        │Single Image │     │Optical + SAR│     │Bi-temporal  │
        │             │     │             │     │             │
        │RSVQA        │     │Native       │     │ChangerEx    │
        │Grounding    │     │Optical/SAR  │     │Change       │
        │SVE          │     │Fusion       │     │Interpreter  │
        │Captioning   │     │             │     │             │
        └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
               │                   │                   │
               └───────────────────┼───────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Evidence Fusion       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │ Answer • Visual Evidence            │
                │ Confidence • Limitations            │
                │ Provenance • Execution Trace        │
                │ Downloadable Report                 │
                └─────────────────────────────────────┘
