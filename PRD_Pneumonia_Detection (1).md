# Product Requirements Document (PRD)
## PneumoScan — A Chest X-Ray Pneumonia Triage System That Knows Its Own Limits

**Version:** 2.0
**Status:** ✅ **Shipped** — built, deployed and documented as of 2026-07-28
**Owner:** Priyanka M
**Target Employer Context:** Fractal Analytics / Qure.ai (medical imaging AI)
**Timeline:** 3 weeks (focused)
**Live Demo:** https://pneumonia-detection-cnn-chi.vercel.app
**API:** https://pneumoscan-api-b76x.onrender.com · **Model:** https://huggingface.co/decstzz06/pneumoscan-model

> **This document is the requirements spec as written before the build, kept intact.**
> Every *required* feature shipped, and the §8.1 scope-cut ladder was **never descended** —
> the build landed at level 1 (full React UI + cloud FastAPI + all four analysis features +
> Grad-CAM), including the Tier-2 Model Report tab that §8.1 listed as the first thing to drop.
> Three **optional** extras were not done: a from-scratch CNN baseline for comparison, the
> README hero GIF, and the blog post.
> Where the spec and the shipped system differ (backend host, serving runtime, chart library),
> the deviations and their reasons are recorded in
> [BUILD_PLAN.md § How it actually went](BUILD_PLAN.md#how-it-actually-went).
> For current state, measured results and limitations, read [README.md](README.md) and
> [MODEL_CARD.md](MODEL_CARD.md) — those are authoritative, not this file.

---

## 0. What Makes This Project Different

Most student portfolios stop at "I trained a CNN on the Kaggle pneumonia dataset and got 95% accuracy." That project has been done thousands of times and signals tutorial-following, not judgment.

**This project is deliberately built around a different, rarer narrative:**

> *"I built a medical model, then stress-tested it the way someone who actually ships clinical AI would — I tested it on data from a different hospital, audited it for hidden bias, checked whether its confidence scores can be trusted, and was honest about where it breaks."*

That theme is backed by real published research: models trained on one chest X-ray dataset routinely lose performance on another, and studies have shown that even *how* a scan was acquired can drive AI performance disparities. Reproducing that finding at a student scale — and being honest about it — is what turns a common project into a memorable one.

Every feature below reinforces this single narrative. Nothing is added just to pad the feature list.

---

## 1. Overview

### 1.1 Problem Statement

Pneumonia is a leading cause of death worldwide, and chest X-rays are the primary diagnostic tool. In many regions radiologist availability is the bottleneck — a scan may sit unread for hours. An automated triage system that flags likely-pneumonia scans for urgent human review, *and honestly flags when it is unsure*, can reduce time-to-diagnosis without pretending to replace a doctor.

### 1.2 What This Is (and Is Not)

**Is:** a decision-support / triage prototype that ranks scans by urgency and abstains when uncertain.

**Is NOT:** an FDA-cleared diagnostic device. It makes no autonomous clinical decisions; every positive or uncertain case routes to a human. Framing it this way is itself a maturity signal — overclaiming medical validity is a red flag to any healthcare-AI employer.

### 1.3 The Core Insight Driving the Design

A model that reports 95% accuracy on its own test set but has never been checked on outside data, never audited for bias, and whose "91% confidence" is meaningless is *not* a trustworthy medical tool. This project treats trustworthiness — generalization, fairness, calibration, and honest uncertainty — as the actual product, with raw accuracy as merely the starting point.

---

## 2. Goals & Non-Goals

### 2.1 Goals

| # | Goal | Why it matters |
|---|------|----------------|
| G1 | Train a CNN classifier via transfer learning (ResNet50 / DenseNet121) | Industry-standard strong baseline |
| G2 | Optimize and report **sensitivity ≥ 0.92** on the internal test set | Missed pneumonia is the costly error |
| G3 | **External validation** on a second, different-source dataset | The headline feature — proves generalization awareness |
| G4 | **Subgroup / fairness audit** across available metadata | Surfaces hidden bias; echoes real published findings |
| G5 | **Calibration analysis + fix** (reliability diagram, temperature scaling) | Makes confidence scores trustworthy for triage |
| G6 | **Uncertainty-aware triage** — abstain & escalate low-confidence cases | Reframes model as a triage tool, Qure.ai's actual product |
| G7 | Grad-CAM interpretability for every prediction | Trust and debuggability |
| G8 | Deploy FastAPI API + React UI to the cloud | A real, clickable product, not a notebook |
| G9 | Full reproducibility + honest model card | Engineering discipline |

### 2.2 Non-Goals

- Multi-class disease detection (only Normal vs Pneumonia)
- Bacterial vs viral subtyping
- DICOM ingestion / mobile app
- Real regulatory compliance or prospective clinical validation

Stating non-goals prevents scope creep in a 3-week window and shows PM-level scoping.

---

## 3. Feature Deep-Dive (the differentiators)

### 3.1 External Validation — the headline

Train on the Kaggle (Paul Mooney) dataset. Then, **without any retraining**, evaluate on a second dataset from a different source — the RSNA / NIH-derived pneumonia challenge subset. Report internal vs external metrics side by side and analyze the drop.

- **Expected outcome:** a visible performance drop (this is normal and is the point).
- **What to show:** a comparison table (internal vs external sensitivity/specificity/AUC) and an honest written explanation of *why* — different scanners, patient populations, labeling methods.
- **Why it lands:** it demonstrates the single thing employers fear juniors don't grasp: models that shine in a notebook often fail on real-world data.

### 3.2 Subgroup / Fairness Audit

Break down performance across whatever metadata is available (e.g., view position AP vs PA, and age bands where present). Report per-subgroup sensitivity and flag any group the model underserves.

- Directly mirrors published research showing acquisition parameters can drive AI disparities.
- **Deliverable:** a small fairness table + 2-3 sentences of honest interpretation.

### 3.3 Calibration Analysis + Fix

A model can be accurate yet badly *calibrated* — its "90% confident" might really be right only 70% of the time. Plot a reliability diagram, compute Expected Calibration Error, then apply **temperature scaling** and show the improvement.

- Almost no student does this. It's exactly what "high-stakes prediction" requires.
- **Deliverable:** before/after reliability diagrams + ECE numbers.

### 3.4 Uncertainty-Aware Triage

Instead of forcing a hard Normal/Pneumonia label, define three zones from the calibrated probability:
- **High-confidence Normal** → routine
- **High-confidence Pneumonia** → flag urgent for radiologist
- **Low-confidence (uncertain band)** → **abstain and escalate** for human review

Report a "coverage vs accuracy" trade-off: accuracy on the cases the model *chose* to decide. This is precisely how real triage AI is deployed.

### 3.5 Grad-CAM Interpretability

Heatmap overlay for every inference, showing where the model looked. Hard requirement — an unexplained medical prediction has little value, and it's a sanity check that the model attends to lungs, not artifacts.

---

## 4. Target Metrics & Acceptance Thresholds

Accuracy alone is misleading (classes are imbalanced ~3:1). Primary metric is **sensitivity**.

| Metric | Internal target | External (expect a drop) |
|--------|-----------------|--------------------------|
| Sensitivity (Recall) | **≥ 0.92** | Report honestly, analyze gap |
| Specificity | **≥ 0.75** | Report honestly |
| ROC-AUC | **≥ 0.94** | Report honestly |
| Expected Calibration Error | Report; reduce via temp. scaling | Report |
| Per-subgroup sensitivity | No group < 0.85 (target) | Report |

**Key design decision:** the classification threshold is **tuned on validation to hit the sensitivity target**, not left at 0.5. A deliberate, defensible clinical trade-off and a strong interview talking point.

The external metrics being *lower* than internal is not a failure of the project — reporting and explaining that gap honestly is the whole point.

---

## 5. Data Requirements

### 5.1 Sources

| Role | Dataset | Notes |
|------|---------|-------|
| Train + internal test | Kaggle Chest X-Ray Pneumonia (Paul Mooney) | ~5,856 images, Normal/Pneumonia |
| **External validation** | RSNA Pneumonia Detection subset (NIH-derived) | Different source, different population |

### 5.2 Known Data Issues (must be handled)

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Kaggle `val/` has only ~16 images | Meaningless validation | **Re-split** into stratified 80/10/10 |
| Class imbalance (~3:1) | Biased model | Class weights + augmentation; PR curve |
| Pediatric-only (Kaggle) vs mixed-age (RSNA) | Drives the external drop | This *is* the story — document it |
| Different labeling methods across datasets | Not perfectly comparable | State explicitly in model card |

Catching the tiny-val-set flaw and the cross-dataset mismatch is what separates you from naive uses of these very popular datasets.

### 5.3 Preprocessing

Resize 224×224; normalize with ImageNet stats; grayscale→3-channel; training augmentation (rotation ±20°, shift 0.2, horizontal flip, zoom 0.2). Apply identical preprocessing to the external set — no peeking, no retraining.

---

## 6. Model Requirements

### 6.1 Architecture

**Primary:** Transfer learning, ResNet50 or DenseNet121 (ImageNet), custom head:
`GlobalAvgPool → Dense(512, ReLU) → BatchNorm → Dropout(0.3) → Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Dense(1, sigmoid)`

**Training:** freeze backbone → train head → unfreeze top blocks → fine-tune at LR 1e-5.

**Optional comparison:** a from-scratch CNN as a baseline. Comparing honestly is a plus.

### 6.2 Config

BCE loss with class weights; Adam (1e-4 → 1e-5); EarlyStopping on `val_auc`, ReduceLROnPlateau, ModelCheckpoint; track accuracy/precision/recall/AUC live.

### 6.3 Post-Training Analysis Pipeline (the differentiators)

A dedicated `evaluate.py` that runs, in order: internal test metrics → external validation → subgroup audit → calibration + temperature scaling → uncertainty-band triage report → Grad-CAM gallery. Each writes a figure/table into `assets/`.

---

## 7. System Architecture

```
┌──────────────┐     HTTPS      ┌─────────────────┐
│  React SPA   │ ─────────────► │  FastAPI service │
│ (Vercel/     │  upload image  │  (HF Spaces /    │
│  Netlify)    │ ◄───────────── │   Render)        │
└──────────────┘  JSON+heatmap  └────────┬─────────┘
   shows: label, calibrated               │
   confidence, uncertainty                ▼
   band, Grad-CAM overlay        ┌──────────────────┐
                                 │ Calibrated model  │
                                 │ + Grad-CAM        │
                                 │ + triage logic    │
                                 └──────────────────┘
```

### 7.1 Components

| Component | Tech | Hosting |
|-----------|------|---------|
| Frontend | React + Vite + Tailwind | Vercel / Netlify (free) |
| Backend API | FastAPI + Uvicorn | HF Spaces / Render (free) |
| Model artifact | Keras SavedModel + temperature param | Hugging Face Hub |
| Code | GitHub (public) | GitHub |

### 7.2 API Contract

**`POST /predict`** — multipart image upload

```json
{
  "prediction": "Pneumonia",
  "triage_zone": "urgent_review",
  "calibrated_confidence": 0.88,
  "is_uncertain": false,
  "probability": { "normal": 0.12, "pneumonia": 0.88 },
  "threshold_used": 0.42,
  "gradcam_overlay": "<base64-png>",
  "recommendation": "Flag for radiologist review",
  "disclaimer": "Research prototype. Not a diagnostic device."
}
```

When `is_uncertain` is true, `triage_zone` becomes `"needs_human_review"` and the UI says so plainly — a feature, not a bug.

**`GET /health`** → `{"status":"ok","model_version":"2.0"}`

### 7.3 Non-Functional Requirements

Inference < 3s on free-tier CPU; graceful 422 on bad uploads; CORS for the frontend; model loaded once at startup; prominent disclaimer in UI and API.

---

## 7.5 Dashboards & Visual Storytelling (the "looks like a real DS project" layer)

### The principle: dashboards must answer questions, not decorate

A common mistake that makes student repos look *weaker* is a wall of colorful charts with no purpose — it reads as "I know how to call `plt.plot()`," not "I think like a data scientist." Every dashboard below exists to answer a specific question a real ML team would ask about this model. That framing is what makes a GitHub repo look professional.

This project ships **three tiers of visual output**, each serving a different audience.

### Tier 1 — The Model Evaluation Dashboard (for technical reviewers)

A single, well-organized dashboard (built in the evaluation notebook and exported as a static image + an interactive HTML version) that tells the model's whole story at a glance. This is the artifact a hiring manager screenshots into a Slack channel.

**Panels, each tied to a narrative beat:**

| Panel | Question it answers | Chart type |
|-------|--------------------|-----------|
| Internal vs External metrics | "Does it generalize?" | Grouped bar (sensitivity/specificity/AUC, side by side) |
| ROC + PR curves (both datasets overlaid) | "How much does performance shift?" | Overlaid curves |
| Confusion matrices (internal & external) | "What kind of errors, and where?" | Annotated heatmaps |
| Calibration reliability diagram (before/after temp. scaling) | "Can I trust its confidence?" | Reliability curve + diagonal |
| Subgroup sensitivity | "Who does it fail?" | Horizontal bar with a target line |
| Coverage vs accuracy (triage) | "How good when it chooses to decide?" | Line: accuracy as abstention rises |

**Design rules that make it look senior, not student:**
- One consistent color palette across every figure (pick two accent colors, use them everywhere)
- Every axis labeled with units; every chart has a one-line takeaway caption *above* it
- A muted, clean theme (e.g. seaborn `whitegrid` or a custom matplotlib style) — no default rainbow, no chartjunk, no 3D pie charts
- Consistent font and figure size so panels align in the README

### Tier 2 — The Live Interactive Dashboard (for the interviewer to click)

Part of the deployed web app — the thing you send as a link. Beyond the single-image upload + prediction, it includes a **"Model Report" tab** that renders the evaluation story interactively:

- Toggle between internal and external results
- Hover tooltips on the curves (Plotly / Recharts)
- A live threshold slider that shows sensitivity/specificity trading off in real time — this one interaction alone signals deep understanding, because it makes the precision/recall trade-off *tangible*
- The Grad-CAM overlay shown next to each uploaded prediction

Interactivity here is purposeful: the threshold slider and dataset toggle let a reviewer *explore the trade-offs you reasoned about*, which is far more persuasive than static claims.

### Tier 3 — The README as a Visual Front Door (the highest-ROI item)

**This is the single most important visual deliverable, and the one students most often neglect.** A recruiter spends 30-60 seconds on your repo before deciding. The README must do the work.

Structure:
1. **One-line pitch** + a **hero image/GIF** at the very top — a short screen recording of the live app making a prediction with a Grad-CAM overlay. A GIF of the working product above the fold is worth more than any paragraph.
2. **Live demo + model card badges** (shields.io) linking out
3. **The headline result stated plainly:** "Sensitivity 0.93 internal → 0.81 external — and here's why," with the internal-vs-external bar chart right there
4. **Architecture diagram** (the one from §7)
5. **Embedded key dashboard panels** (calibration, fairness, triage) with one-sentence takeaways
6. **Reproducibility block** — how to run it in 3 commands

A README that opens with a working-product GIF, a headline generalization result, and clean embedded charts is what "looks exactly like a real data-scientist project" actually means.

### Tooling

- **Static dashboards:** matplotlib + seaborn with a shared custom style file (`src/viz_style.py`) so every figure is visually consistent — consistency is the tell of a professional
- **Interactive web dashboard:** Plotly (Python side) or Recharts (React side)
- **README GIF:** any screen recorder → export as optimized GIF
- **Badges:** shields.io

### Scope note

The Tier-1 static dashboard and a strong README (Tier 3) are **non-negotiable and cheap** — build them regardless. The Tier-2 interactive "Model Report" tab (especially the threshold slider) is high-impact but optional under time pressure; it slots into the scope-cut ladder at level 2.

---

## 8. Deployment Plan

- **Model** → Hugging Face Hub (free, versioned, resume-worthy link)
- **Backend** → HF Spaces (Docker) or Render free tier
- **Frontend** → Vercel (auto-deploys on push)

### 8.1 Scope-Cut Ladder (if you fall behind)

Cut from the bottom; the interviewer never sees what you cut.

1. **Full build:** React UI + FastAPI on cloud + all four analysis features + Grad-CAM *(target)*
2. If tight: keep the Tier-1 dashboard + README front door (cheap, high-impact); drop the Tier-2 interactive Model Report tab and use a minimal React page
3. If tighter: FastAPI + Gradio/Streamlit on HF Spaces (still cloud, still clickable)
4. Floor: FastAPI `/docs` (Swagger) as the demo + a strong README with screenshots

**Feature priority if time is short:** keep External Validation (3.1) and Uncertainty Triage (3.4) at all costs — they carry the narrative. Calibration (3.3) and the subgroup audit (3.2) are the next to trim, but each is only ~half a day and adds a lot.

---

## 9. Timeline (3 Weeks)

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Data + model | Re-split, augmentation, trained model hitting internal targets, saved best model |
| **Week 2** | The differentiators + API | External validation, subgroup audit, calibration, uncertainty triage, Grad-CAM; FastAPI working; backend deployed |
| **Week 3** | Frontend + dashboards + polish | React UI, **Tier-1 evaluation dashboard**, **Tier-2 Model Report tab**, deploy to Vercel, **README as visual front door (hero GIF, embedded charts)**, model card, blog post |

Front-load model-hitting-targets in Week 1. The differentiators in Week 2 are where the resume value is — protect that week.

---

## 10. Deliverables (Portfolio Artifacts)

1. **Public GitHub repo** — clean structure, pinned deps, README with architecture diagram
2. **Live demo URL** — the clickable app
3. **Model card** — dataset, metrics, **external results, fairness, calibration**, intended use, limitations
4. **Generalization report** — internal vs external metrics + honest analysis (the centerpiece)
5. **Model Evaluation Dashboard** — the single-image Tier-1 dashboard (see §7.5) that tells the whole model story
6. **Interactive "Model Report" tab** in the live app (Tier-2), incl. the live threshold slider
7. **A README built as a visual front door** — hero GIF, headline result, embedded charts (see §7.5, Tier 3)
8. **Fairness table**, **calibration diagrams**, **coverage-accuracy triage curve** (also embedded in the dashboard)
9. **Grad-CAM gallery**
10. **Notebooks** — EDA, training, evaluation (reproducible)
11. **Blog post** — *"My medical AI model looked great — until I tested it on a different hospital's data."* This title alone signals maturity.

### 10.1 Repo Structure

```
pneumoscan/
├── README.md
├── MODEL_CARD.md
├── requirements.txt
├── data/            # download + re-split scripts
├── notebooks/       # EDA, training, evaluation
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py       # external val, fairness, calibration, triage
│   ├── calibration.py
│   ├── gradcam.py
│   ├── viz_style.py      # shared matplotlib/seaborn style — consistent figures
│   └── dashboard.py      # builds the Tier-1 evaluation dashboard
├── api/             # FastAPI app + Dockerfile
├── frontend/        # React app
└── assets/          # curves, tables, gradcam samples, screenshots
```

---

## 11. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Model misses internal targets | Medium | Fine-tune backbone; tune threshold; report honestly |
| External drop is large/embarrassing | Medium | **That's the finding** — explain it; it's the strongest part |
| 3-week timeline slips | Med-High | Scope-cut ladder (§8.1); protect the Week-2 differentiators |
| Free-tier hosting sleeps | Medium | Accept cold starts; "warming up" UI state; note in README |
| External dataset access/format friction | Medium | Pick the RSNA subset early in Week 1; script the download |
| Interviewer questions medical validity | Low | Lean into "triage prototype, not device" — the correct answer |

---

## 12. Interview Talking Points

Each maps to what Fractal / Qure.ai cares about:

- **External validation** — "My model dropped from X to Y on outside data, and here's why" (the strongest thing you can say as a junior)
- **Fairness audit** — you thought about *who* the model fails
- **Calibration** — the difference between a confident model and a trustworthy one
- **Uncertainty triage** — knowing when *not* to predict
- **Recall over accuracy** + **threshold tuning** — asymmetric cost of a missed diagnosis
- **Grad-CAM** — explainability
- **The tiny-val-set catch** — you found a real data flaw
- **The evaluation dashboard & threshold slider** — you can *show*, live, how sensitivity and specificity trade off, instead of just describing it
- **Keywords:** Qure.ai, medical imaging, generalization, distribution shift, calibration, fairness, sensitivity, interpretability, triage, high-stakes predictions

---

## 13. Future Work (shows vision)

- DICOM ingestion instead of JPEG/PNG
- Multi-label detection (effusion, cardiomegaly, etc.)
- Deep-ensemble or MC-dropout uncertainty instead of threshold bands
- Validation on a third dataset (PadChest / CheXpert) for a stronger generalization claim
- Production model-monitoring / drift detection
