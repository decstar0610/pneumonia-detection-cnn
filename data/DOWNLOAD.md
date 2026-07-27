# Data Download Guide

Two datasets. Drop each into the folder shown and the code in `src/config.py` finds it
automatically. Verify anytime with:

```bash
python -m src.data     # prints READY / MISSING for both datasets
```

---

## Prerequisite: Kaggle API (one-time)

Both datasets come from Kaggle, so set up the CLI once.

```bash
pip install kaggle
```

Then get an API token: **kaggle.com → your profile → Settings → API → "Create New Token"**.
This downloads `kaggle.json`. Place it where the CLI looks:

- **Windows (PowerShell):**
  ```powershell
  New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
  Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json" -Force
  ```
- **macOS / Linux:**
  ```bash
  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
  ```

Verify: `kaggle --version`.

---

## 1. Kaggle Chest X-Ray Pneumonia  (train + internal test)

**Source:** `paultimothymooney/chest-xray-pneumonia` (~5,856 images, ~2 GB).
**Target folder:** `data/raw/kaggle_chest_xray/`

### Download command

**Windows (PowerShell):**
```powershell
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p .\data\raw\kaggle_chest_xray --unzip
```

**macOS / Linux (bash):**
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p ./data/raw/kaggle_chest_xray --unzip
```

> If `--unzip` leaves a nested `chest_xray/chest_xray/`, that's fine — but the expected layout
> below must resolve. If it double-nests, move the inner `chest_xray/` up one level.

### Expected layout when done
```
data/raw/kaggle_chest_xray/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/       *.jpeg
    │   └── PNEUMONIA/    *.jpeg
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── val/              # only ~16 images — we RE-SPLIT in Phase 1 (do not trust as-is)
        ├── NORMAL/
        └── PNEUMONIA/
```

> Note: the tiny `val/` is a known flaw (PRD §5.2). Phase 1 pools everything and re-splits
> stratified 80/10/10 into `data/splits/splits.csv`. You still download it as-is.

---

## 2. RSNA Pneumonia Detection Challenge  (external validation)

**Source:** competition `rsna-pneumonia-detection-challenge` (NIH-derived, DICOM, ~3.5 GB).
**Target folder:** `data/raw/rsna/`

### Step A — accept the rules (required, one-time)
Open **https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules** and click
**"I Understand and Accept"**. Competition downloads 403 until you do this.

### Step B — download command

**Windows (PowerShell):**
```powershell
kaggle competitions download -c rsna-pneumonia-detection-challenge -p .\data\raw\rsna
Expand-Archive .\data\raw\rsna\rsna-pneumonia-detection-challenge.zip -DestinationPath .\data\raw\rsna -Force
```

**macOS / Linux (bash):**
```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge -p ./data/raw/rsna
unzip -o ./data/raw/rsna/rsna-pneumonia-detection-challenge.zip -d ./data/raw/rsna
```

### Expected layout when done
```
data/raw/rsna/
├── stage_2_train_images/            *.dcm  (DICOM)
├── stage_2_train_labels.csv         patientId, x, y, width, height, Target
└── stage_2_detailed_class_info.csv  patientId, class
```

> We use RSNA **only for external validation** — no retraining, identical preprocessing
> (PRD §3.1). `Target` (0/1) becomes NORMAL/PNEUMONIA in Phase 4.1. The `class` column and
> DICOM header (e.g. ViewPosition AP/PA) feed the fairness audit in Phase 4.4.

### Lighter alternative (if 3.5 GB is too heavy right now)
You only need a validation subset. Options:
- Download the full set and sample a few thousand images in Phase 4.1, **or**
- Use the smaller RSNA-derived PNG mirror on Kaggle and adjust `RSNA_IMAGES` in `src/config.py`.
Either way the external-validation hook (`src/data.load_external_rsna`) stays the same.

---

## After both are in place
```bash
python -m src.data
# Kaggle (train/internal): READY  -> ...\data\raw\kaggle_chest_xray\chest_xray
# RSNA   (external valid): READY  -> ...\data\raw\rsna\stage_2_train_images
```
Then Phase 1 (re-split + EDA) can run. Nothing else references dataset paths directly —
change a location once in `src/config.py` and everything follows.
