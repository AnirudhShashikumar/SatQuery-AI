<div align="center">

# 🛰️ SatQuery AI

### Ask Earth Anything. Get Evidence, Not Just Answers.

**An evidence-first, agentic Vision-Language platform for multimodal Earth-observation and remote-sensing analysis.**

<br/>

Natural-Language Query  
↓  
Satellite Observation(s)  
↓  
Scientific Validation  
↓  
Agentic Specialist Selection  
↓  
Remote-Sensing AI  
↓  
**Answer + Visual Evidence + Confidence + Provenance + Execution Trace**

<br/>

**Smart India Hackathon 2026 · ISRO / Department of Space**

**Problem Statement 26167 · Space Technology · Software**

<br/>

[Overview](#-overview) ·
[Why SatQuery](#-why-satquery-ai) ·
[Capabilities](#-core-capabilities) ·
[Architecture](#-system-architecture) ·
[AI Specialists](#-ai-specialists) ·
[Benchmarks](#-verified-benchmark-results) ·
[Scientific Guardrails](#-scientific-guardrails) ·
[Installation](#-getting-started) ·
[Research](#-research--datasets)

</div>

---

# 🌍 Overview

Satellite imagery contains enormous amounts of information about the Earth, but extracting that information remains difficult.

Analysts often need to understand sensor characteristics, preprocess geospatial data, select task-specific models, interpret outputs, compare multiple observations, and manually validate results before reaching a conclusion.

**SatQuery AI turns that workflow into a natural-language interaction.**

A user can upload Earth-observation imagery and ask questions such as:

> **“Describe the land-cover and major objects visible in this image.”**

> **“Highlight the water body referred to in the query.”**

> **“Use the optical and SAR images together to identify built-up and water-covered regions.”**

> **“What changed between these two dates, and where did the change occur?”**

SatQuery does not simply send the image and prompt to one general-purpose Vision-Language Model.

Instead, it:

1. validates the supplied observations,
2. determines their modality and compatibility,
3. interprets the user's intent,
4. selects appropriate remote-sensing specialists,
5. executes the required scientific workflow,
6. fuses the resulting evidence,
7. estimates confidence,
8. exposes limitations and provenance,
9. and returns both the answer **and the evidence supporting it**.

---

# 🎯 Why SatQuery AI?

Most AI assistants are designed around:

```text
Image + Prompt → Model → Answer
```

SatQuery is designed around:

```text
Observation(s)
      +
Natural-Language Question
      │
      ▼
Scientific Input Validation
      │
      ▼
Task & Modality Understanding
      │
      ▼
Agentic Specialist Selection
      │
      ▼
Remote-Sensing Analysis
      │
      ▼
Evidence Generation
      │
      ▼
Evidence Fusion
      │
      ▼
Answer
+ Spatial Evidence
+ Confidence
+ Provenance
+ Limitations
+ Execution Trace
```

### The central idea

> **A remote-sensing AI system should not only tell the user what it believes.  
> It should expose what evidence caused it to reach that conclusion.**

SatQuery is therefore built around four principles:

| Principle | Meaning |
|---|---|
| 🛰️ **Remote-Sensing Native** | Designed around optical, multispectral, SAR and temporal EO observations |
| 🧠 **Multi-Specialist** | Different tasks are handled by specialized models rather than one generic model |
| 🤖 **Agentic** | The system determines which scientific workflow and specialist should answer the query |
| 🔬 **Evidence-First** | Answers are accompanied by evidence, provenance, limitations and execution information |

---

# ✨ Core Capabilities

SatQuery currently supports three primary Earth-observation analysis modes plus dedicated spatial grounding.

---

## 🖼️ 1. Single Image Analysis

Analyze an individual:

- Optical image
- Multispectral observation
- SAR observation

Supported tasks include:

### Remote-Sensing VQA

Ask natural-language questions about an observation.

```text
What type of land cover dominates this image?
```

```text
Are there buildings visible in the scene?
```

```text
What major objects are present?
```

### Scene Understanding

Generate structured remote-sensing scene interpretation.

### Text-Guided Grounding

Locate regions or objects described through language.

```text
Find the road.
```

```text
Highlight the water body.
```

### Remote-Sensing Representation Analysis

SatQuery's adapted visual encoder can provide scene-level remote-sensing evidence and semantic priors.

---

# 🛰️ 2. Optical + SAR Analysis

Optical and Synthetic Aperture Radar sensors observe different physical characteristics of the Earth's surface.

SatQuery combines their evidence without pretending that the two modalities are interchangeable.

```text
Optical Observation ─────┐
                         │
                         ▼
                    Evidence Fusion
                         ▲
                         │
SAR Observation ─────────┘
```

The workflow can support:

- built-up-region analysis,
- water-region analysis,
- optical/SAR agreement,
- complementary sensor evidence,
- cross-modal reasoning,
- modality-specific interpretation,
- qualitative fusion,
- and scientifically constrained spatial comparison.

Example:

> **“Use the optical and SAR images together to identify built-up and water-covered regions.”**

### Native evidence remains separate

The optical observation and native SAR observation are independently processed.

This matters because SAR is not simply a grayscale optical image.

SatQuery preserves modality provenance throughout the workflow.

---

# 🕒 3. Bi-Temporal Analysis

SatQuery can analyze spatially corresponding observations captured at two different times.

```text
Earlier Observation
        │
        ├───────────────┐
        │               │
        ▼               ▼
   Change Model    Supporting Evidence
        │               │
        └───────┬───────┘
                ▼
       Semantic Interpreter
                │
                ▼
     Change Answer + Evidence
```

Supported functionality includes:

- learned change detection,
- spatial change localization,
- changed-area estimation,
- connected change regions,
- change descriptions,
- change-oriented VQA,
- scene-level semantic comparison,
- deterministic supporting evidence,
- and confidence-aware interpretation.

Example:

> **“What changed between these two dates, and where did the change occur?”**

Another example:

> **“Has the built-up area increased, decreased, or remained unchanged?”**

---

# 🎯 4. Text-Guided Spatial Grounding

SatQuery supports open-vocabulary localization of remote-sensing objects and regions.

Examples:

```text
Find the road.
```

```text
Locate the aircraft.
```

```text
Highlight the water body.
```

Instead of returning only text, grounding produces **spatial visual evidence**.

---

# 🧠 Agentic Remote-Sensing Intelligence

The SatQuery Agent acts as an orchestration layer between the user and the specialist models.

Its job is not to replace the specialists.

Its job is to determine **which specialist should be trusted for which task**.

The agent performs operations including:

```text
Query Interpretation
        ↓
Input Validation
        ↓
Modality Detection
        ↓
Observation Role Validation
        ↓
Task Classification
        ↓
Compatibility Analysis
        ↓
Specialist Selection
        ↓
Parameter Validation
        ↓
Workflow Execution
        ↓
Evidence Fusion
        ↓
Confidence / Limitations
        ↓
Response Generation
```

This allows the platform to support multiple scientific workflows through one natural-language interface.

---

# 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                           │
│                                                             │
│              Natural Language + EO Observation(s)           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                SCIENTIFIC INPUT VALIDATOR                   │
│                                                             │
│  Format · Modality · Bands · CRS · Grid · Metadata          │
│  Registration · Resolution · Observation Roles              │
│  Spatial Compatibility · SAR Characteristics                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     SATQUERY AGENT                          │
│                                                             │
│       Intent Classification · Routing · Configuration       │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
┌──────────────────┐ ┌────────────────┐ ┌───────────────────┐
│   SINGLE IMAGE   │ │ OPTICAL + SAR  │ │   BI-TEMPORAL     │
│                  │ │                │ │                   │
│ RSVQA            │ │ Native Optical │ │ ChangerEx         │
│ Grounding        │ │ Native SAR     │ │ Change Analyzer   │
│ SVE              │ │ Cross-Modal    │ │ Semantic          │
│ Captioning       │ │ Fusion         │ │ Interpreter       │
└─────────┬────────┘ └───────┬────────┘ └─────────┬─────────┘
          │                  │                    │
          └──────────────────┼────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      EVIDENCE FUSION                        │
│                                                             │
│ Spatial Evidence · Scene Evidence · Native Sensor Evidence  │
│ Agreement · Disagreement · Provenance · Limitations         │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       FINAL RESULT                          │
│                                                             │
│ Answer                                                      │
│ Visual Evidence                                             │
│ Confidence                                                  │
│ Scientific Limitations                                      │
│ Model / Tool Provenance                                     │
│ Execution Timeline                                          │
│ Downloadable Report                                         │
└─────────────────────────────────────────────────────────────┘
```

---

# 🤖 AI Specialists

SatQuery uses a **specialist registry** rather than relying on one monolithic model.

---

## 🧠 RSVQA Specialist v1

Purpose:

**Remote-Sensing Visual Question Answering**

Designed to answer questions about individual Earth-observation scenes.

The specialist combines remote-sensing visual representations with question understanding and task-specific answer heads.

---

## 🎯 Grounding Specialist v1.1

Purpose:

**Natural-language spatial grounding**

Built around an open-vocabulary grounding pipeline and adapted/evaluated for remote-sensing imagery.

Returns spatial regions corresponding to textual queries.

---

## 🌍 SatQuery Vision Encoder — SVE

Purpose:

**Remote-sensing visual representation**

SVE is based on an **OpenCLIP ViT-L/14-derived visual encoder** adapted using remote-sensing data.

It contributes:

- scene-level embeddings,
- semantic scene priors,
- representation evidence,
- supporting multimodal information.

### Important

SVE scene similarities are **not calibrated probabilities**.

SVE does not by itself provide:

- ground truth,
- pixel segmentation,
- object localization,
- or calibrated class probabilities.

---

## 🛰️ Native SAR Analysis

Purpose:

**Preserve and analyze SAR evidence without pretending SAR is optical imagery.**

SatQuery examines characteristics such as:

- channel count,
- polarization information,
- numeric value domain,
- representation type,
- visualization strategy.

---

## 🔀 Cross-Modal Optical + SAR Fusion

Purpose:

**Combine complementary evidence from optical and SAR observations.**

Evidence maintains explicit provenance:

```text
optical
native_sar
fused
```

This prevents generated or derived representations from silently replacing authoritative source observations.

---

## 🕒 ChangerEx

Purpose:

**Learned bi-temporal change detection**

SatQuery uses a local PyTorch reconstruction of ChangerEx based on the Open-CD implementation.

The official checkpoint is strict-loaded and the reconstructed implementation has undergone parity validation.

ChangerEx provides spatial change evidence used by the higher-level change interpretation workflow.

---

## 🔎 Deterministic Change Analyzer

Purpose:

**Independent supporting change evidence**

The deterministic analyzer provides additional evidence that can be compared against learned change detection.

It can also provide a fallback when learned evidence is unavailable.

---

## 🎨 Pix2Pix

Purpose:

**SAR → optical-like interpretive visualization**

Pix2Pix can produce an RGB-like learned representation of SAR imagery.

This representation may help downstream visual interpretation.

However:

> **Generated RGB is not observed optical imagery.**

The original SAR observation remains authoritative.

---

## 🧪 SARFusionFormer

Purpose:

**Experimental SAR → optical translation**

SARFusionFormer is a custom multi-scale CNN + transformer architecture developed for SAR-to-optical research.

It expects explicit SAR polarization channels such as:

```text
VV
VH
```

The system does **not** silently duplicate channels to satisfy model requirements.

SARFusionFormer remains an experimental supporting model rather than an authoritative optical source.

---

# 🧬 Remote-Sensing Adaptation

A major design requirement of SatQuery is that its visual intelligence should not rely exclusively on generic natural-image representations.

The **SatQuery Vision Encoder** was adapted using remote-sensing imagery and text derived from BigEarthNet resources.

Adaptation resources include:

### BigEarthNet

Remote-sensing imagery used to adapt the visual representation.

### BigEarthNet.txt

Large-scale text descriptions associated with remote-sensing scenes.

The adaptation pipeline used matched remote-sensing image/text pairs to move the representation toward the Earth-observation domain.

---

# 📊 Verified Benchmark Results

SatQuery intentionally separates:

```text
MODEL RUNS
      ≠
BENCHMARK ACCURACY
      ≠
SYSTEM READINESS
```

Only measured results are reported as benchmark performance.

---

## RSVQA Specialist v1

**Dataset:** RSVQA-LR  
**Split:** Official Test  
**Questions:** 10,004

| Metric | Result |
|---|---:|
| Exact Accuracy | **71.05%** |
| Overflow-Aware Accuracy | **74.31%** |

---

## Grounding Specialist v1.1

**Dataset:** VRSBench  
**Split:** Full Validation  
**Samples:** 16,146

| Metric | Result |
|---|---:|
| Mean IoU | **0.2348** |
| Accuracy @ IoU 0.50 | **25.05%** |

Grounding remains one of SatQuery's weakest quantitatively measured specialists.

That limitation is intentionally reported rather than hidden behind selected visual examples.

---

# 🧪 Evaluation Infrastructure

The repository contains dedicated evaluation tooling for major SatQuery workflows.

```text
scripts/
├── evaluate_cdvqa.py
├── evaluate_changerex_levircd.py
├── evaluate_optical_sar.py
├── evaluate_sar_translation.py
├── evaluate_captions.py
├── calibrate_confidence.py
└── benchmark_performance.py
```

Evaluation infrastructure distinguishes:

- benchmark evidence,
- implementation readiness,
- runtime health,
- missing evaluation evidence.

A benchmark that has not been executed is reported as **unavailable**, not as `0`.

---

# 🔬 Scientific Guardrails

Scientific constraints are treated as product features rather than implementation details.

---

## 1. No Verified Registration → No Verified Pixel-Level Claims

Two images having identical width and height does **not** prove that they are spatially registered.

For geospatial observations SatQuery evaluates information including:

- Coordinate Reference System
- affine transform
- spatial extent
- pixel resolution
- orientation
- overlap
- grid correspondence
- NoData compatibility

Possible compatibility states include:

```text
EXACT_GRID_MATCH
SAME_AREA_DIFFERENT_GRID
REPROJECTION_REQUIRED
RESAMPLING_REQUIRED
PARTIAL_OVERLAP
INSUFFICIENT_OVERLAP
UNVERIFIABLE
```

Quantitative spatial claims are restricted when the required registration cannot be established.

---

## 2. SAR → RGB ≠ Optical Truth

SAR-to-optical translation is useful as an **interpretive visualization**.

It does not reconstruct an authoritative optical observation.

Therefore:

```text
Generated RGB
    ≠
Observed Optical Image
    ≠
Ground Truth
```

Generated representations are explicitly identified through provenance.

---

## 3. Change Mask ≠ Semantic Change

A binary change detector can provide evidence that:

> **something changed here**

It does not independently prove:

> **what changed or why**

For example, a binary mask alone cannot establish:

- vegetation loss,
- new construction,
- flooding,
- demolition,
- built-up increase,
- agricultural conversion.

Those interpretations require additional evidence.

---

## 4. Scene Similarity ≠ Probability

Embedding similarity from SVE represents semantic similarity.

It should not be interpreted as a calibrated class probability.

---

## 5. Native Evidence Remains Authoritative

Derived products do not silently replace original observations.

Examples include:

```text
Original SAR
    ↓
Generated RGB

Original SAR remains available and authoritative.
```

---

## 6. Model Failure Must Not Become Evidence

If an evidence-producing model fails, SatQuery prevents its failed output from silently appearing as successful evidence.

Generation state, semantic interpretation state and evidence provenance remain distinct.

---

# 🛰️ SAR-Aware Processing

SAR requires different treatment from optical imagery.

SatQuery includes deterministic value-domain inspection capable of distinguishing inputs such as:

```text
integer-like
dB-like
amplitude/power-like
unknown floating point
```

Where available, the pipeline also records:

- data type,
- channel count,
- polarization,
- display stretch,
- interpretation reasoning.

Models requiring VV/VH polarization must receive explicit compatible inputs.

SatQuery does not silently invent missing polarizations.

---

# 🌈 Multispectral Awareness

For multispectral inputs SatQuery records:

- number of bands,
- available band names,
- selected visualization bands,
- selection rationale.

Known mappings such as Sentinel-like:

```text
B4 → Red
B3 → Green
B2 → Blue
```

can be used when appropriate.

Unknown 4+ band observations are not automatically treated as RGB without justification.

---

# 🧾 Evidence & Provenance

Every specialist can publish evidence into the analysis record.

Evidence may contain:

```text
Evidence Source
Specialist
Model
Input Observation
Observation Role
Evidence Type
Spatial Region
Confidence
Parameters
Execution State
Limitations
```

The result interface can then distinguish between:

```text
Authoritative source imagery
Learned spatial evidence
Generated supporting imagery
Scene-level semantic evidence
Deterministic evidence
Fused evidence
```

---

# 🔍 Execution Trace

SatQuery exposes an observable execution trace.

A typical analysis may include:

```text
VALIDATION
│
├── Format Validation
├── Modality Inspection
├── Observation Role Validation
├── Geospatial Compatibility
└── Specialist Eligibility

ROUTING
│
├── Query Classification
└── Specialist Selection

MODEL INFERENCE
│
├── Image Preparation
├── Specialist Execution
└── Evidence Generation

EVIDENCE FUSION
│
├── Evidence Validation
├── Agreement Analysis
└── Confidence Assessment

RESPONSE
│
├── Scientific Interpretation
└── Final Answer
```

This makes the execution auditable without requiring exposure of private internal model reasoning.

---

# 🖥️ Evidence-First User Interface

SatQuery includes a complete interactive research workspace.

Major sections include:

### Analysis

- Single Image
- Optical + SAR
- Bi-temporal
- Grounding
- Mission & Sensor Comparison

### Research Workspace

- Reports
- Benchmarks
- Demo Gallery
- Research Analytics
- SIH Compliance

### Model Inspection

- SatQuery Vision Encoder
- Pix2Pix
- SARFusionFormer
- Change Detection
- Color Correction

### Developer / System

- API
- Logs
- Settings
- Health

The analysis experience exposes:

- source observations,
- detected modality,
- effective modality,
- workflow routing,
- evidence viewer,
- specialist evidence,
- confidence,
- runtime,
- execution timeline,
- scientific limitations,
- PDF export,
- JSON export,
- complete ZIP export,
- stored-result comparison.

---

# 📑 Reports & Exports

SatQuery analyses can be exported for inspection and reproducibility.

Supported result experiences include:

```text
PDF Report
JSON Record
Complete ZIP
Stored-Result Comparison
```

Exports are generated from the authoritative analysis record so that the report and interface remain aligned with the executed workflow.

---

# 💻 Local-First Execution

SatQuery's core architecture is designed to operate locally where compatible model checkpoints are available.

Supported compute paths include:

```text
Apple Silicon → MPS
NVIDIA GPU    → CUDA
Fallback      → CPU
```

This enables the core scientific pipeline to remain independent of mandatory external generative-AI APIs.

Optional external providers may be used for supporting language-generation functionality, but they are not intended to replace the underlying remote-sensing specialists.

---

# 🛠️ Technology Stack

## AI / Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)

- Python
- PyTorch
- OpenCLIP
- Grounding DINO
- Open-CD / ChangerEx
- Computer Vision
- Vision-Language Models
- Representation Learning

## Remote Sensing / Geospatial

- Rasterio
- GeoTIFF
- Sentinel-1 SAR
- Sentinel-2 Optical
- Multispectral imagery
- CRS / transform validation
- spatial grid analysis
- multimodal EO processing

## Backend

![FastAPI](https://img.shields.io/badge/FastAPI-Agent%20Backend-009688?logo=fastapi)

- FastAPI
- Uvicorn
- Typed API contracts
- Agentic orchestration
- Specialist registry
- Evidence lifecycle
- Report generation

## Frontend

![Next.js](https://img.shields.io/badge/Next.js-Application-black?logo=nextdotjs)
![React](https://img.shields.io/badge/React-UI-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-Type%20Safety-3178C6?logo=typescript)

- Next.js
- React
- TypeScript
- Tailwind CSS
- Vitest

---

# 📁 Repository Structure

A simplified view of the project:

```text
sar-colorization-app/
│
├── backend.py
│
├── satquery_agent/
│   ├── orchestration
│   ├── specialists
│   ├── validation
│   ├── evidence
│   └── workflows
│
├── models/
│   ├── SARFusionFormer
│   ├── Pix2Pix
│   ├── Change Detection
│   └── supporting models
│
├── datasets/
│
├── scripts/
│   ├── evaluation
│   ├── benchmarking
│   └── calibration
│
├── tests/
│
├── docs/
│
└── frontend/
    │
    ├── app/
    │
    ├── components/
    │   ├── satquery/
    │   ├── viewer/
    │   └── benchmarks/
    │
    ├── hooks/
    ├── lib/
    ├── services/
    ├── types/
    └── public/
```

---

# 🚀 Getting Started

> **Note:** Model checkpoints and large datasets may not be distributed directly with this repository. Follow the relevant model and dataset licenses.

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd sar-colorization-app
```

---

## 2. Create a Python virtual environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start the SatQuery backend

```bash
python3 -m uvicorn backend:app \
  --host 127.0.0.1 \
  --port 8010
```

Check system health:

```text
http://127.0.0.1:8010/health
```

---

## 5. Install frontend dependencies

Open another terminal:

```bash
cd frontend
npm install
```

---

## 6. Configure frontend environment

Create:

```text
frontend/.env.local
```

Example:

```env
NEXT_PUBLIC_SATQUERY_API_URL=http://127.0.0.1:8010
```

> **Never commit `.env.local` or API credentials to Git.**

---

## 7. Start the frontend

```bash
npm run dev
```

Then open the local URL shown by Next.js.

---

# 🍎 macOS Application

SatQuery can also be packaged as a native-style macOS launcher.

The launcher can:

1. start the FastAPI backend,
2. wait for the health endpoint,
3. start the production frontend,
4. open SatQuery in the browser,
5. reuse already-running SatQuery services.

This provides a double-click workflow for local demonstrations while retaining the Python + Next.js architecture underneath.

---

# 🧪 Testing

Backend and frontend functionality are protected through automated testing.

Typical checks include:

```bash
pytest
```

and:

```bash
cd frontend
npm test
```

Additional release checks include:

```text
TypeScript validation
ESLint
Next.js production build
Problem-statement regression tests
Workflow smoke tests
Model/checkpoint verification
Submission audit
```

Test counts evolve as the project develops, so the repository's current CI/test output should be treated as authoritative.

---

# 📚 Research & Datasets

SatQuery builds upon open research in remote sensing, computer vision and vision-language intelligence.

---

## BigEarthNet

Used for remote-sensing visual representation adaptation.

---

## RSVQA

Used for remote-sensing Visual Question Answering evaluation.

Verified SatQuery evaluation:

```text
Official test questions: 10,004
Exact accuracy:          71.05%
Overflow-aware:          74.31%
```

---

## VRSBench

Used for remote-sensing grounding evaluation.

Verified SatQuery evaluation:

```text
Validation samples: 16,146
Mean IoU:           0.2348
Acc@0.50:           25.05%
```

---

## SEN12MS-CR

Used in SAR → optical research.

Provides paired Sentinel-1 SAR and Sentinel-2 observations.

---

## LEVIR-CD

Used as a change-detection evaluation target for the bi-temporal pipeline.

---

## CDVQA

Used as an evaluation target for change-oriented Visual Question Answering.

---

## RSICD

Used in remote-sensing image-captioning research.

---

# 📖 Model & Research Foundations

SatQuery builds upon or integrates ideas and implementations from research including:

### OpenCLIP

Used as the foundation for SatQuery's remote-sensing adapted visual representation.

### Grounding DINO

Used for open-vocabulary visual grounding.

### ChangerEx / Open-CD

Used for learned bi-temporal change detection.

### Pix2Pix

Used for experimental SAR-to-optical-like visualization.

These projects remain the work of their respective authors and organizations.

Please cite the original research when using SatQuery in academic work.

---

# 🏆 Smart India Hackathon 2026

SatQuery AI was developed for:

| | |
|---|---|
| **Organization** | ISRO — Department of Space |
| **Problem Statement ID** | **26167** |
| **Category** | Software |
| **Theme** | Space Technology |
| **Project** | SatQuery AI |
| **Team** | Code Blue |

The challenge focuses on building an interactive Vision-Language assistant for multimodal remote-sensing imagery through text queries.

SatQuery addresses the challenge through:

```text
                     SATQUERY AI
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
     SINGLE IMAGE    OPTICAL + SAR    BI-TEMPORAL
          │               │               │
     VQA              Fusion          Change Detection
     Captioning       Analysis        Change Description
     Grounding                        Change VQA
```

The platform also implements remote-sensing adaptation and agentic specialist orchestration.

---

# 🔐 Scientific Honesty

SatQuery follows one rule throughout the project:

> **Never claim more than the evidence supports.**

Therefore:

- smoke tests are not reported as accuracy benchmarks,
- missing benchmark results are not converted into zeroes,
- generated SAR imagery is not called optical ground truth,
- scene similarity is not presented as probability,
- unregistered imagery is not treated as pixel-aligned,
- binary change masks are not automatically assigned semantic causes,
- model confidence is not called calibrated probability unless calibration has been performed,
- hidden-dataset accuracy is never claimed before evaluation.

---

# ⚠️ Current Research Limitations

SatQuery is an active research platform.

Important remaining evaluation areas include:

### Grounding

Current measured grounding performance leaves substantial room for improvement.

### Change Detection

Full benchmark evaluation remains an important validation priority.

### Change VQA

CDVQA evaluation should be completed before claiming benchmark-level change-VQA performance.

### Optical + SAR

Controlled quantitative evaluation of cross-modal fusion remains a priority.

### Sensor Domain Shift

Performance on unseen satellite/sensor distributions requires further evaluation.

In particular, results on hidden evaluation imagery should not be predicted from public benchmark performance.

### Confidence Calibration

Not every specialist confidence value represents a formally calibrated probability.

---

# 🛣️ Roadmap

### Implemented

- [x] Natural-language EO interface
- [x] Agentic task routing
- [x] Scientific input validation
- [x] Optical imagery support
- [x] Multispectral-aware processing
- [x] Native SAR processing
- [x] Single-image VQA
- [x] Text-guided grounding
- [x] Remote-sensing adapted vision encoder
- [x] Optical + SAR workflow
- [x] Cross-modal evidence fusion
- [x] Learned bi-temporal change detection
- [x] Semantic change interpretation
- [x] Deterministic supporting change evidence
- [x] Evidence provenance
- [x] Scientific limitations
- [x] Execution traces
- [x] Evidence viewer
- [x] Reports
- [x] JSON exports
- [x] Result comparison
- [x] Benchmark dashboard
- [x] Research analytics
- [x] SIH compliance dashboard
- [x] Local Apple Silicon inference
- [x] macOS launcher

### Research / Evaluation Priorities

- [ ] Extended CDVQA evaluation
- [ ] Full LEVIR-CD evaluation
- [ ] Controlled Optical + SAR benchmark
- [ ] Additional captioning evaluation
- [ ] Grounding improvements
- [ ] Confidence calibration
- [ ] Cross-sensor robustness evaluation
- [ ] Additional multispectral support

---

# 🎬 Example Workflow

### User

```text
What changed between these two dates, and where did the change occur?
```

### SatQuery

```text
1. Detects two observations
2. Assigns earlier/later temporal roles
3. Validates compatibility
4. Classifies query as CHANGE_VQA
5. Checks change-specialist eligibility
6. Executes learned change detection
7. Generates spatial change evidence
8. Computes supporting evidence
9. Interprets only evidence-supported changes
10. Returns answer + evidence + confidence + limitations
```

The result is therefore not simply:

```text
"Buildings increased."
```

Instead, SatQuery attempts to provide:

```text
ANSWER
What the available evidence supports.

SPATIAL EVIDENCE
Where the model detected change.

CONFIDENCE
How strongly the available evidence supports the result.

LIMITATIONS
What cannot safely be concluded.

PROVENANCE
Which model/tool produced each piece of evidence.

EXECUTION TRACE
How the workflow was performed.
```

---

# 🌎 Potential Applications

SatQuery's architecture is relevant to areas including:

### 🌊 Disaster Response

Rapid inspection of changing or inaccessible regions using multimodal observations.

### 🏙️ Urban Monitoring

Analyze built-up development and spatial change.

### 🌾 Agriculture

Explore agricultural landscapes and temporal variation.

### 🌳 Environmental Monitoring

Inspect land-cover patterns and environmental change.

### 💧 Water Resources

Identify and analyze water-related regions across observations.

### 🛰️ Remote-Sensing Research

Provide a unified interface for testing and inspecting specialist EO models.

---

# 🔭 Vision

Earth-observation systems are becoming increasingly multimodal.

A single region may be observed using:

- optical sensors,
- multispectral sensors,
- Synthetic Aperture Radar,
- multiple acquisition dates,
- and eventually additional geospatial modalities.

The challenge is therefore no longer simply:

> **Can a model understand an image?**

The more important question is:

> **Can an intelligent system determine which observations, models and evidence are appropriate for answering a scientific question — and show the user why the resulting answer should be trusted?**

That is the problem SatQuery AI is designed to explore.

---

# 👥 Team

## Code Blue

Developed for **Smart India Hackathon 2026**.

SatQuery AI combines:

```text
Remote Sensing
      +
Computer Vision
      +
Vision-Language Intelligence
      +
Geospatial Processing
      +
Agentic AI
      +
Evidence-Based Reasoning
```

into a unified Earth-observation analysis platform.

---

# 🤝 Contributing

SatQuery AI is currently under active research and development.

When contributing, please preserve the project's core scientific principles:

1. Do not silently coerce incompatible sensor data.
2. Preserve source-observation provenance.
3. Distinguish generated evidence from authoritative observations.
4. Do not fabricate missing benchmark results.
5. Do not treat model execution as proof of model accuracy.
6. Make uncertainty and limitations visible.
7. Add tests for changes affecting scientific workflow behavior.

---

# 🔒 Security

Never commit:

```text
.env.local
API keys
provider secrets
service credentials
private model tokens
private datasets
restricted ISRO evaluation data
```

Use environment variables and local configuration for secrets.

---

# 📜 Disclaimer

**SatQuery AI is a research and decision-support platform.**

Its outputs — including generated imagery, VQA answers, object locations, change masks, semantic interpretations and confidence estimates — should not automatically be treated as authoritative geospatial ground truth.

High-impact operational decisions should be independently validated using appropriate Earth-observation data, domain expertise and established analytical procedures.

---

# ⭐ Support the Project

If you find SatQuery AI interesting, consider starring the repository.

It helps make the project more visible to researchers, developers and others interested in multimodal Earth-observation intelligence.

---

<div align="center">

# 🛰️ SatQuery AI

### Ask Earth Anything.

## Get Evidence, Not Just Answers.

**Remote-Sensing Intelligence · Agentic AI · Multimodal Evidence · Scientific Transparency**

<br/>

Built by **Team Code Blue** for  
**Smart India Hackathon 2026 · ISRO / Department of Space**

</div>
