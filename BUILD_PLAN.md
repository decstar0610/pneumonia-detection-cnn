# PneumoScan — Phased Build Plan

A concrete, phase-by-phase execution plan derived from `PRD_Pneumonia_Detection (1).md`.
Each phase lists: **objective**, **tasks**, **files produced**, **deliverables**, and **exit criteria**
(you don't leave a phase until its exit criteria are met).

**Guiding principle from the PRD:** the product is *trustworthiness* (generalization, fairness,
calibration, honest uncertainty), not raw accuracy. Protect the differentiator work (Phase 4).

**Primary metric:** Sensitivity (Recall) ≥ 0.92 internal. Accuracy is misleading (~3:1 imbalance).

---

## Phase map at a glance

| Phase | Name | PRD refs | Week | Priority |
|-------|------|----------|------|----------|
| 0 | Project setup & scaffolding | §10.1 | Wk1 | Must |
| 1 | Data pipeline & EDA | §5 | Wk1 | Must |
| 2 | Model & training | §6.1–6.2 | Wk1 | Must |
| 3 | Internal evaluation & threshold tuning | §4 | Wk1→2 | Must |
| 4 | The differentiators | §3, §6.3 | Wk2 | **Protect** |
| 5 | Tier-1 evaluation dashboard | §7.5 | Wk2→3 | Must |
| 6 | FastAPI backend | §7.2–7.3 | Wk2 | Must |
| 7 | React frontend + Tier-2 report | §7.5 | Wk3 | High/Opt |
| 8 | Deployment | §8 | Wk3 | Must |
| 9 | Docs: README, model card, blog | §7.5, §10 | Wk3 | Must |

Scope-cut ladder (from §8.1) is honored in each phase's "if behind" note.

---

## Phase 0 — Project setup & scaffolding

**Objective:** a clean, reproducible skeleton so every later phase drops into a known place.

**Tasks**
- Create repo structure per PRD §10.1 (`src/`, `api/`, `frontend/`, `notebooks/`, `data/`, `assets/`).
- Set up Python env: `requirements.txt` (tensorflow/keras, scikit-learn, pandas, numpy,
  matplotlib, seaborn, opencv, fastapi, uvicorn, pillow, netcal/temperature-scaling helpers, tqdm).
- Pin versions; add `.gitignore` (exclude `data/raw`, model weights, `__pycache__`, node_modules).
- `git init`, first commit. Decide GPU vs CPU training (confirm what's available on this machine).
- Config module (`src/config.py`) — all paths, image size (224), batch size, seeds, thresholds in one place.
- `src/viz_style.py` stub — shared matplotlib/seaborn style (two accent colors, whitegrid).

**Files produced:** `requirements.txt`, `.gitignore`, `src/config.py`, `src/viz_style.py`, folder tree.

**Exit criteria:** `python -c "import tensorflow, sklearn, fastapi"` works in the env; repo committed.

**If behind:** none — this is cheap and non-negotiable.

---

## Phase 1 — Data pipeline & EDA

**Objective:** trustworthy splits and a clear picture of the data (incl. the known flaws from §5.2).

**Tasks**
- Point the pipeline at the **local Kaggle dataset** (path TBD — you'll provide it).
- **Re-split fix (critical, §5.2):** the Kaggle `val/` has only ~16 images. Merge all data and
  re-split into **stratified 80/10/10** train/val/test with a fixed seed. Save the split manifest
  (CSV of filepath→label→split) so it's reproducible and auditable.
- Preprocessing (§5.3): resize 224×224, ImageNet normalization, grayscale→3-channel.
  Training augmentation: rotation ±20°, shift 0.2, horizontal flip, zoom 0.2. **No aug on val/test.**
- Build `tf.data` (or Keras) generators with class weighting for the ~3:1 imbalance.
- **EDA notebook** (`notebooks/01_eda.ipynb`): class counts, image size distribution, sample grid,
  imbalance ratio, and explicit documentation of the tiny-val flaw + pediatric-only nature (this
  becomes part of the model-card story).

**Files produced:** `data/resplit.py`, `src/data.py` (loaders/preprocess), `notebooks/01_eda.ipynb`,
`data/splits.csv` (manifest).

**Exit criteria:** stratified split reproducible from seed; generators yield correct shapes/labels;
EDA notebook documents imbalance + the two known data issues.

**If behind:** none — the re-split is core to the narrative.

---

## Phase 2 — Model & training

**Objective:** a trained transfer-learning model that hits internal targets, saved and reproducible.

**Tasks**
- `src/model.py`: transfer learning backbone (**DenseNet121 or ResNet50**, ImageNet weights) + custom head:
  `GlobalAvgPool → Dense(512,ReLU) → BN → Dropout(0.3) → Dense(256,ReLU) → BN → Dropout(0.3) → Dense(1,sigmoid)`.
- `src/train.py`: two-stage schedule —
  1. freeze backbone, train head (Adam 1e-4);
  2. unfreeze top blocks, fine-tune (Adam 1e-5).
- Loss: BCE with class weights. Callbacks: EarlyStopping on `val_auc`, ReduceLROnPlateau,
  ModelCheckpoint (best by val_auc). Track accuracy/precision/recall/AUC live.
- Save best model (Keras SavedModel) + training history plot.
- **Optional:** a small from-scratch CNN baseline for an honest comparison (scope-cut candidate).

**Files produced:** `src/model.py`, `src/train.py`, `models/best_model/` (or HF-bound artifact),
`assets/training_history.png`.

**Exit criteria:** model trains end-to-end; **internal test sensitivity ≥ 0.92, specificity ≥ 0.75,
AUC ≥ 0.94** (or a documented gap + plan). Best model saved and reloadable.

**If behind:** drop the from-scratch baseline; keep transfer-learning model only.

---

## Phase 3 — Internal evaluation & threshold tuning

**Objective:** honest internal metrics and a **deliberately tuned threshold** (not 0.5).

**Tasks**
- `src/evaluate.py` (part 1): internal test metrics — confusion matrix, sensitivity, specificity,
  precision, ROC-AUC, PR-AUC.
- **Threshold tuning (§4 key decision):** choose the operating threshold **on the validation set**
  to hit sensitivity ≥ 0.92, then report the resulting specificity. Persist `threshold_used`.
- ROC + PR curves saved to `assets/`.

**Files produced:** `src/evaluate.py` (internal section), `assets/roc_pr_internal.png`,
`assets/confusion_internal.png`, `threshold.json`.

**Exit criteria:** internal metrics table produced; tuned threshold persisted and justified in writing.

**If behind:** none — this feeds everything downstream.

---

## Phase 4 — The differentiators  ⭐ (protect this phase)

**Objective:** the four trust features + Grad-CAM. This is the resume value. Order matters.

### 4.1 External validation (the headline, §3.1)
- Wire in the **RSNA/NIH external set** (path TBD; may need a small download/prep script + labels CSV).
- Apply **identical preprocessing**, **no retraining**, **no peeking**.
- Produce internal-vs-external side-by-side table (sensitivity/specificity/AUC) + ROC/PR overlays.
- Write the honest "why it dropped" analysis (scanners, populations pediatric-vs-mixed-age, labeling).

### 4.2 Calibration + fix (§3.3)
- `src/calibration.py`: reliability diagram + Expected Calibration Error (ECE).
- Fit **temperature scaling** on validation; show before/after reliability + ECE improvement.
- Save the temperature parameter alongside the model (used at inference).

### 4.3 Uncertainty-aware triage (§3.4)
- From calibrated probabilities define 3 zones: high-conf Normal (routine),
  high-conf Pneumonia (urgent), uncertain band (**abstain & escalate**).
- Produce the **coverage-vs-accuracy** curve (accuracy on cases it chose to decide).

### 4.4 Subgroup / fairness audit (§3.2)
- Per-subgroup sensitivity across available metadata (view position AP/PA, age bands where present).
- Fairness table + flag any group < 0.85; 2–3 sentences honest interpretation.

### 4.5 Grad-CAM (§3.5)
- `src/gradcam.py`: heatmap overlay for any image; sanity-check the model attends to lungs.
- Build a Grad-CAM gallery for `assets/`.

**Files produced:** `src/evaluate.py` (external/fairness/triage sections), `src/calibration.py`,
`src/gradcam.py`, `assets/` figures + tables (internal-vs-external, reliability before/after,
coverage-accuracy, fairness table, gradcam gallery), `temperature.json`.

**Exit criteria:** all five artifacts exist and are honest; external drop is measured and explained;
temperature scaling shows an ECE reduction; triage zones + coverage curve produced.

**If behind (§8.1 priority):** keep **External Validation (4.1)** and **Uncertainty Triage (4.3)**
at all costs. Calibration (4.2) and Fairness (4.4) are next to trim (each ~½ day). Grad-CAM stays
(it's a hard requirement and cheap).

---

## Phase 5 — Tier-1 evaluation dashboard

**Objective:** one senior-looking dashboard that tells the whole model story at a glance (§7.5 Tier 1).

**Tasks**
- `src/dashboard.py`: assemble 6 panels — internal-vs-external bars, ROC+PR overlay, confusion
  matrices, calibration before/after, subgroup sensitivity (with target line), coverage-vs-accuracy.
- Enforce design rules: one 2-accent palette everywhere, labeled axes/units, one-line takeaway
  caption above each panel, muted whitegrid theme, consistent font/figure size.
- Export a static PNG (for README) **and** an interactive HTML version.

**Files produced:** `src/dashboard.py`, `assets/evaluation_dashboard.png`, `assets/dashboard.html`,
`notebooks/03_evaluation.ipynb`.

**Exit criteria:** single dashboard image renders all six panels consistently; screenshot-ready.

**If behind:** static PNG is non-negotiable; the interactive HTML can slip.

---

## Phase 6 — FastAPI backend

**Objective:** a real, callable service implementing the §7.2 contract.

**Tasks**
- `api/main.py`: load calibrated model + temperature + threshold **once at startup**.
- `POST /predict` (multipart image) → JSON exactly per §7.2: `prediction`, `triage_zone`,
  `calibrated_confidence`, `is_uncertain`, `probability{normal,pneumonia}`, `threshold_used`,
  `gradcam_overlay` (base64 PNG), `recommendation`, `disclaimer`.
- `is_uncertain=true` ⇒ `triage_zone="needs_human_review"`.
- `GET /health` → `{"status":"ok","model_version":"2.0"}`.
- NFRs (§7.3): inference < 3s on CPU, graceful **422** on bad uploads, CORS for frontend,
  prominent disclaimer. `Dockerfile` for HF Spaces/Render.

**Files produced:** `api/main.py`, `api/inference.py`, `api/Dockerfile`, `api/requirements.txt`.

**Exit criteria:** `/predict` returns the full contract with a real Grad-CAM overlay locally;
`/health` ok; bad upload → 422; CORS set.

**If behind:** this is the demo floor (§8.1 level 4 = Swagger `/docs`) — keep it working.

---

## Phase 7 — React frontend + Tier-2 Model Report

**Objective:** the clickable product (§7.5 Tier 2).

**Tasks**
- React + Vite + Tailwind. Upload image → show label, calibrated confidence, uncertainty band,
  Grad-CAM overlay, and a prominent disclaimer.
- **Model Report tab (Tier 2, optional under pressure):** internal/external toggle, Plotly/Recharts
  curves with hover, and the **live threshold slider** (sensitivity/specificity trade off in real time).

**Files produced:** `frontend/` (Vite app), components for upload + report tab.

**Exit criteria:** upload→prediction works against the deployed/local API; disclaimer visible.

**If behind (§8.1):** level 2 = minimal React page, drop Model Report tab; level 3 = Gradio/Streamlit
on HF Spaces instead of React.

---

## Phase 8 — Deployment

**Objective:** everything live and linkable (§8).

**Tasks**
- Push model artifact + temperature param to **Hugging Face Hub** (versioned link).
- Backend → **HF Spaces (Docker)** or **Render** free tier.
- Frontend → **Vercel** (auto-deploy on push).
- Handle free-tier cold starts: "warming up" UI state; note it in README.
- Smoke-test the deployed `/predict` end to end.

**Files produced:** deploy configs, live URLs.

**Exit criteria:** public demo URL works end-to-end; model on HF Hub; health check green.

**If behind (§8.1):** fall back down the ladder (Gradio on Spaces → Swagger `/docs` + README screenshots).

---

## Phase 9 — Docs: README front door, model card, blog

**Objective:** the highest-ROI visual work — a recruiter decides in 30–60s (§7.5 Tier 3).

**Tasks**
- **README as visual front door:** one-line pitch + **hero GIF** (screen recording of a live
  prediction with Grad-CAM) above the fold; shields.io badges + live demo link; the headline result
  stated plainly ("Sensitivity 0.93 internal → 0.81 external — and here's why") with the
  internal-vs-external bar chart; architecture diagram (§7); embedded calibration/fairness/triage
  panels with one-sentence takeaways; a 3-command reproducibility block.
- **MODEL_CARD.md:** datasets, metrics, **external results, fairness, calibration**, intended use,
  limitations, the pediatric-vs-mixed-age caveat, "triage prototype, not a device."
- **Generalization report** (the centerpiece) — internal vs external + honest analysis.
- **Blog post:** *"My medical AI model looked great — until I tested it on a different hospital's data."*

**Files produced:** `README.md`, `MODEL_CARD.md`, generalization report, blog draft, `assets/hero.gif`.

**Exit criteria:** README opens with working-product GIF + headline result + clean embedded charts;
model card honest and complete.

**If behind:** README front door + model card are non-negotiable; blog post can trail.

---

## Cross-cutting: reproducibility & honesty (throughout)
- Fixed seeds everywhere; split manifest committed; pinned deps.
- Every figure/table written to `assets/` by a script (not hand-made), so results regenerate.
- Never let external metrics "look better" — the honest gap **is** the deliverable (§11).
- Keep the framing disciplined: triage decision-support, **not** an FDA diagnostic device.

## Immediate next action
Provide the **local Kaggle dataset path** (and RSNA path if you have it) so Phase 1 can start.
Then I'll execute Phase 0 (scaffold) and Phase 1 (re-split + EDA) first.
