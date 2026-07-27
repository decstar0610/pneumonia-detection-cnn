"""Data hooks — stubbed for Kaggle (train/internal) and RSNA (external validation).

Nothing here trains a model. These functions:
  1. Validate that the datasets were dropped into the expected folders (per data/DOWNLOAD.md).
  2. Give a precise, actionable error if they are missing (pointing at the download command).
  3. Expose the loader entry points the rest of the pipeline (Phase 1+) will fill in.

Run `python -m src.data` to check whether the data is in place.
"""
from __future__ import annotations

from pathlib import Path

from . import config as C


class DatasetNotReady(FileNotFoundError):
    """Raised when a dataset folder is missing or malformed. Message tells you the fix."""


# --- Structure validation ----------------------------------------------------

def _kaggle_expected_tree() -> list[Path]:
    """The subfolders the Kaggle set must expose after extraction."""
    tree: list[Path] = []
    for split in ("train", "test", "val"):
        for cls in C.CLASSES:
            tree.append(C.KAGGLE_IMAGES / split / cls)
    return tree


def check_kaggle(strict: bool = True) -> bool:
    """Verify the Kaggle Chest X-Ray Pneumonia folder structure exists.

    Expected (after extraction):
        data/raw/kaggle_chest_xray/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/*.jpeg
    """
    missing = [p for p in _kaggle_expected_tree() if not p.is_dir()]
    if missing:
        if strict:
            raise DatasetNotReady(
                "Kaggle Chest X-Ray dataset not found / incomplete.\n"
                f"  Expected image root: {C.KAGGLE_IMAGES}\n"
                f"  Missing folders: {[str(p) for p in missing[:4]]}"
                f"{' ...' if len(missing) > 4 else ''}\n"
                "  Fix: see data/DOWNLOAD.md → 'Kaggle Chest X-Ray Pneumonia'.\n"
                "  Quick: `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia "
                f"-p {C.KAGGLE_ROOT} --unzip`"
            )
        return False
    return True


def check_rsna(strict: bool = True) -> bool:
    """Verify the RSNA external validation set exists (images + labels CSV)."""
    problems: list[str] = []
    if not C.RSNA_IMAGES.is_dir():
        problems.append(f"missing image dir: {C.RSNA_IMAGES}")
    if not C.RSNA_LABELS_CSV.is_file():
        problems.append(f"missing labels csv: {C.RSNA_LABELS_CSV}")
    if problems:
        if strict:
            raise DatasetNotReady(
                "RSNA external validation dataset not found / incomplete.\n"
                f"  {'; '.join(problems)}\n"
                "  Fix: see data/DOWNLOAD.md → 'RSNA Pneumonia Detection Challenge'.\n"
                "  Quick: accept the competition rules on Kaggle, then\n"
                "  `kaggle competitions download -c rsna-pneumonia-detection-challenge "
                f"-p {C.RSNA_ROOT} && (unzip into {C.RSNA_ROOT})`"
            )
        return False
    return True


# --- Loader entry points ------------------------------------------------------

def load_kaggle_manifest():
    """Return the stratified 80/10/10 split manifest (`data/splits/splits.csv`) as a DataFrame.

    Built by `data/resplit.py`, which pools all Kaggle images and re-splits with a fixed seed
    (fixes the ~16-image val/ flaw, PRD §5.2). Everything downstream reads this, never the
    original Kaggle folder split.
    """
    import pandas as pd

    if not C.SPLIT_MANIFEST.is_file():
        check_kaggle(strict=True)  # surfaces the download hint if data itself is missing
        raise DatasetNotReady(
            f"Split manifest not found: {C.SPLIT_MANIFEST}\n"
            "  Fix: run `python data/resplit.py` to build the stratified 80/10/10 split."
        )
    return pd.read_csv(C.SPLIT_MANIFEST)


# ImageNet normalization constants, shaped for broadcasting over an HxWx3 image.
def _norm_constants():
    import numpy as np

    mean = np.array(C.IMAGENET_MEAN, dtype="float32").reshape(1, 1, 3)
    std = np.array(C.IMAGENET_STD, dtype="float32").reshape(1, 1, 3)
    return mean, std


def _decode_and_preprocess(path):
    """Read an image file → 224x224x3 float tensor, [0,1]-scaled then ImageNet-standardized.

    grayscale→3-channel is handled by decoding with channels=3. Applied identically to
    train/val/test AND the external set (no peeking).
    """
    import tensorflow as tf

    mean, std = _norm_constants()
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=C.CHANNELS, expand_animations=False)
    img = tf.image.resize(img, [C.IMG_SIZE, C.IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - tf.constant(mean)) / tf.constant(std)
    img.set_shape((C.IMG_SIZE, C.IMG_SIZE, C.CHANNELS))
    return img


_augmenter = None


def _get_augmenter():
    """Training-only augmentation per PRD §5.3: rotation ±20°, shift 0.2, zoom 0.2, hflip."""
    global _augmenter
    if _augmenter is None:
        import tensorflow as tf
        from tensorflow.keras import layers

        _augmenter = tf.keras.Sequential(
            [
                layers.RandomRotation(20.0 / 360.0),      # ±20 degrees
                layers.RandomTranslation(0.2, 0.2),       # shift 0.2
                layers.RandomZoom(0.2),                   # zoom 0.2
                layers.RandomFlip("horizontal"),          # per PRD (explicitly listed)
            ],
            name="augment",
        )
    return _augmenter


def make_generators(split: str, shuffle: bool | None = None, augment: bool | None = None):
    """Return a batched, prefetched `tf.data.Dataset` of (image, label) for the given split.

    split: 'train' | 'val' | 'test'. Defaults: train shuffles + augments; val/test do neither.
    """
    import tensorflow as tf

    if split not in C.SPLIT_RATIOS:
        raise ValueError(f"split must be one of {tuple(C.SPLIT_RATIOS)}, got {split!r}")
    shuffle = (split == "train") if shuffle is None else shuffle
    augment = (split == "train") if augment is None else augment

    df = load_kaggle_manifest()
    sub = df[df["split"] == split]
    if sub.empty:
        raise RuntimeError(f"No rows for split={split!r} in {C.SPLIT_MANIFEST}.")

    paths = sub["filepath"].to_numpy()
    labels = sub["target"].to_numpy().astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(sub), seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_decode_and_preprocess(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(C.BATCH_SIZE)
    if augment:
        aug = _get_augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def compute_class_weights() -> dict[int, float]:
    """Balanced class weights from the TRAIN split, for BCE loss (~3:1 imbalance, PRD §6.2)."""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    df = load_kaggle_manifest()
    y = df.loc[df["split"] == "train", "target"].to_numpy()
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _rsna_metadata(sample_size: int | None, seed: int):
    """Per-patient RSNA table (patientId, Target, class), optionally stratified-sampled by Target.

    The labels CSV has one row per bounding box (positives repeat), so we dedupe to one row per
    patient. `Target` (0/1) is the external NORMAL/PNEUMONIA label; `class` distinguishes the two
    kinds of negative (true Normal vs "No Lung Opacity / Not Normal" = other pathology) for the
    subgroup audit.
    """
    import pandas as pd

    lab = pd.read_csv(C.RSNA_LABELS_CSV)[["patientId", "Target"]].drop_duplicates("patientId")
    cls = pd.read_csv(C.RSNA_CLASS_INFO_CSV).drop_duplicates("patientId")
    meta = lab.merge(cls, on="patientId", how="left").reset_index(drop=True)
    if sample_size and sample_size < len(meta):
        n = len(meta)
        meta = (
            meta.groupby("Target", group_keys=False)
            .apply(lambda g: g.sample(n=max(1, round(sample_size * len(g) / n)), random_state=seed))
            .reset_index(drop=True)
        )
    return meta


def load_external_rsna(sample_size: int | None = 3000, seed: int = C.SEED):
    """External validation set (RSNA) as a batched `tf.data.Dataset` + aligned metadata DataFrame.

    Returns `(ds, meta)` where `ds` yields (image, label) with the SAME preprocessing as internal
    (resize 224 -> /255 -> ImageNet-standardise; RSNA DICOMs are 8-bit MONOCHROME2, matching the
    Kaggle JPEG convention), and `meta` has columns [patientId, Target, class, view, sex, age].
    No retraining, no threshold re-tuning on this set (Phase 4 uses the internal-tuned threshold).

    `sample_size` stratified-samples by Target for a tractable CPU run (None = all 26,684).
    """
    import numpy as np
    import pydicom
    import tensorflow as tf

    check_rsna(strict=True)
    meta = _rsna_metadata(sample_size, seed)

    # Header-only pre-pass for fairness metadata (view / sex / age) — cheap, no pixel decode.
    views, sexes, ages = [], [], []
    for pid in meta["patientId"]:
        h = pydicom.dcmread(str(C.RSNA_IMAGES / f"{pid}.dcm"), stop_before_pixels=True)
        views.append(str(h.get("ViewPosition", "") or ""))
        sexes.append(str(h.get("PatientSex", "") or ""))
        age_raw = str(h.get("PatientAge", "") or "").rstrip("Y")
        ages.append(int(age_raw) if age_raw.isdigit() else np.nan)
    meta = meta.assign(view=views, sex=sexes, age=ages)

    paths = [str(C.RSNA_IMAGES / f"{pid}.dcm") for pid in meta["patientId"]]
    labels = meta["Target"].to_numpy().astype("float32")
    mean, std = _norm_constants()

    def gen():
        for p, y in zip(paths, labels):
            arr = pydicom.dcmread(p).pixel_array  # HxW uint8, MONOCHROME2
            yield np.repeat(arr[..., None], 3, axis=-1).astype("float32"), y

    out_sig = (
        tf.TensorSpec(shape=(None, None, C.CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    def _prep(img, y):
        img = tf.image.resize(img, [C.IMG_SIZE, C.IMG_SIZE])
        img = img / 255.0
        img = (img - tf.constant(mean)) / tf.constant(std)
        img.set_shape((C.IMG_SIZE, C.IMG_SIZE, C.CHANNELS))
        return img, y

    ds = (
        tf.data.Dataset.from_generator(gen, output_signature=out_sig)
        .map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(C.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds, meta


# --- CLI status check --------------------------------------------------------

if __name__ == "__main__":
    print("Dataset readiness check")
    print("-" * 40)
    kaggle_ok = check_kaggle(strict=False)
    rsna_ok = check_rsna(strict=False)
    print(f"  Kaggle (train/internal): {'READY' if kaggle_ok else 'MISSING'}  -> {C.KAGGLE_IMAGES}")
    print(f"  RSNA   (external valid): {'READY' if rsna_ok else 'MISSING'}  -> {C.RSNA_IMAGES}")
    if not (kaggle_ok and rsna_ok):
        print("\nSee data/DOWNLOAD.md for the exact download commands.")
