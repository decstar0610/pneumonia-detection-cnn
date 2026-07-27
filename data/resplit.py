"""Phase 1 — Stratified re-split of the Kaggle Chest X-Ray Pneumonia dataset.

Fixes the known flaw (PRD §5.2): Kaggle's `val/` has only ~16 images, which is useless for
model selection or threshold tuning. This pools ALL images (train + val + test) and re-splits
into a stratified 80/10/10 train/val/test with a fixed seed, writing a reproducible manifest
to `data/splits/splits.csv`. Every downstream step reads that manifest — never the original
Kaggle folder split.

Run (from repo root, with the venv active):
    python data/resplit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# allow `python data/resplit.py` to import the src package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C            # noqa: E402
from src.data import check_kaggle      # noqa: E402

IMG_EXTS = {".jpeg", ".jpg", ".png"}


def collect_images() -> pd.DataFrame:
    """Pool every image across the original train/val/test folders into one table."""
    rows = []
    for split in ("train", "val", "test"):
        for cls in C.CLASSES:
            folder = C.KAGGLE_IMAGES / split / cls
            if not folder.is_dir():
                continue
            for p in folder.iterdir():
                if p.suffix.lower() in IMG_EXTS:
                    rows.append(
                        {
                            "filepath": str(p.resolve()),
                            "label": cls,
                            "target": 1 if cls == C.POSITIVE_CLASS else 0,
                            "orig_split": split,
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {C.KAGGLE_IMAGES}. Check the download.")
    return df


def stratified_resplit(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a new stratified 80/10/10 split, preserving class balance in each split."""
    r = C.SPLIT_RATIOS
    # first carve out train; then split the remainder into val/test proportionally
    train_df, rest_df = train_test_split(
        df, train_size=r["train"], stratify=df["target"], random_state=C.SEED
    )
    val_frac_of_rest = r["val"] / (r["val"] + r["test"])
    val_df, test_df = train_test_split(
        rest_df, train_size=val_frac_of_rest, stratify=rest_df["target"], random_state=C.SEED
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def summarize(df: pd.DataFrame) -> None:
    print("\nStratified re-split summary")
    print("-" * 48)
    table = (
        df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(["train", "val", "test"])
    )
    table["total"] = table.sum(axis=1)
    table["pneumonia_%"] = (table.get(C.POSITIVE_CLASS, 0) / table["total"] * 100).round(1)
    print(table.to_string())
    print(f"\nTotal images: {len(df)}   seed: {C.SEED}   ratios: {C.SPLIT_RATIOS}")


def main() -> None:
    check_kaggle(strict=True)  # actionable error if the Kaggle data isn't in place
    df = collect_images()
    df = stratified_resplit(df)
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["filepath", "label", "target", "split", "orig_split"]
    df[cols].to_csv(C.SPLIT_MANIFEST, index=False)
    summarize(df)
    print(f"\nWrote manifest -> {C.SPLIT_MANIFEST}")


if __name__ == "__main__":
    main()
