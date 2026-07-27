"""Central configuration: all paths, image params, seeds, and thresholds in one place.

Import from here everywhere so nothing hardcodes a path. Drop the datasets into the
folders described in `data/DOWNLOAD.md` and these paths resolve automatically.
"""
from __future__ import annotations

from pathlib import Path

# --- Repo layout -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"

# --- Dataset roots (where you extract the downloads) -------------------------
# Kaggle "Chest X-Ray Pneumonia" (Paul Mooney). After extraction the dataset ships
# its own `chest_xray/` folder, so the images live one level below KAGGLE_ROOT.
KAGGLE_ROOT = RAW_DIR / "kaggle_chest_xray"
KAGGLE_IMAGES = KAGGLE_ROOT / "chest_xray"  # contains train/ test/ val/

# RSNA Pneumonia Detection Challenge (NIH-derived) — the external validation set.
RSNA_ROOT = RAW_DIR / "rsna"
RSNA_IMAGES = RSNA_ROOT / "stage_2_train_images"          # DICOMs
RSNA_LABELS_CSV = RSNA_ROOT / "stage_2_train_labels.csv"  # patientId, Target, ...
RSNA_CLASS_INFO_CSV = RSNA_ROOT / "stage_2_detailed_class_info.csv"

# Re-split manifest (Phase 1 writes this; everything else reads it).
SPLIT_MANIFEST = SPLITS_DIR / "splits.csv"

# --- Image / training params -------------------------------------------------
IMG_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 32
SEED = 42

# ImageNet normalization stats (grayscale X-rays are stacked to 3 channels).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Stratified re-split ratios (fixes the ~16-image Kaggle val/ flaw, PRD §5.2).
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

# Class label convention.
CLASSES = ("NORMAL", "PNEUMONIA")
POSITIVE_CLASS = "PNEUMONIA"  # the costly-to-miss class; sensitivity target applies here

# --- Decision thresholds (Phase 3/4 persist the tuned values) ----------------
DEFAULT_THRESHOLD = 0.5           # placeholder; tuned on val to hit sensitivity >= 0.92
UNCERTAIN_BAND = (0.40, 0.60)     # calibrated-prob band that abstains & escalates (Phase 4.3)
