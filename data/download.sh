#!/usr/bin/env bash
# One-shot dataset download (macOS / Linux / Git Bash).
# Prereq: `pip install kaggle` and ~/.kaggle/kaggle.json (see data/DOWNLOAD.md).
# For RSNA you must first accept the rules at:
#   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules
#
# Usage (from repo root):
#   ./data/download.sh            # both datasets
#   ./data/download.sh kaggle
#   ./data/download.sh rsna
set -euo pipefail

only="${1:-both}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kaggle_dir="$root/data/raw/kaggle_chest_xray"
rsna_dir="$root/data/raw/rsna"

if [[ "$only" == "both" || "$only" == "kaggle" ]]; then
  echo "==> Kaggle Chest X-Ray Pneumonia -> $kaggle_dir"
  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p "$kaggle_dir" --unzip
fi

if [[ "$only" == "both" || "$only" == "rsna" ]]; then
  echo "==> RSNA Pneumonia Detection Challenge -> $rsna_dir"
  echo "    (must have accepted the competition rules on kaggle.com first)"
  kaggle competitions download -c rsna-pneumonia-detection-challenge -p "$rsna_dir"
  unzip -o "$rsna_dir/rsna-pneumonia-detection-challenge.zip" -d "$rsna_dir"
fi

echo ""
echo "Done. Verify with:  python -m src.data"
