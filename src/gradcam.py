"""Phase 4.5 — Grad-CAM explainability (PRD §3.5 / BUILD_PLAN 4.5).

Grad-CAM shows WHERE the model looks: it weights the last conv feature maps by the gradient of
the prediction w.r.t. those maps, giving a coarse heatmap of the regions that drove the score.
A trustworthy pneumonia model should attend to lung fields, not text markers, borders, or tubes.

Nested-model note: our DenseNet121 backbone is a sub-model inside the full model, so its internal
activations are not exposed in the outer graph. We therefore run a `conv_model`
(backbone.input -> backbone.output, i.e. the final 7x7x1024 'relu' map) and re-apply the head
layers by hand under a GradientTape that watches the conv activations.

Run:
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m src.gradcam        # builds the gallery
"""
from __future__ import annotations

from . import config as C


def _build_cam_fns(model):
    """Return (conv_model, head_layers): the feature extractor and the ordered head to re-apply."""
    import tensorflow as tf

    from .model import get_backbone

    backbone = get_backbone(model)
    conv_model = tf.keras.Model(backbone.input, backbone.output, name="conv_model")
    head_layers = model.layers[model.layers.index(backbone) + 1:]  # gap -> ... -> prob
    return conv_model, head_layers


def compute_heatmap(conv_model, head_layers, img_tensor):
    """Grad-CAM heatmap (224x224, [0,1]) for a single preprocessed image tensor (224x224x3)."""
    import numpy as np
    import tensorflow as tf

    x = tf.convert_to_tensor(img_tensor[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv = conv_model(x, training=False)          # (1,7,7,1024)
        tape.watch(conv)
        h = conv
        for layer in head_layers:
            h = layer(h, training=False)
        score = h[:, 0]                                # sigmoid prob of PNEUMONIA
    grads = tape.gradient(score, conv)                 # (1,7,7,1024)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))    # (1024,) global-avg-pooled gradients
    cam = tf.reduce_sum(conv[0] * weights, axis=-1)    # (7,7)
    cam = tf.nn.relu(cam).numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)                      # normalise to [0,1]
    cam = tf.image.resize(cam[..., None], [C.IMG_SIZE, C.IMG_SIZE]).numpy()[..., 0]
    return float(score.numpy()[0]), cam


def _display_image(path):
    """Original X-ray resized to 224 (grayscale, [0,1]) for overlay display."""
    import tensorflow as tf

    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)
    img = tf.image.resize(img, [C.IMG_SIZE, C.IMG_SIZE]) / 255.0
    return img.numpy()[..., 0]


def _overlay(ax, gray, cam):
    ax.imshow(gray, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.40)
    ax.axis("off")


def build_gallery(n_per_cat: int = 4, seed: int = C.SEED) -> None:
    """Grad-CAM gallery of TP/TN/FP/FN internal-test examples -> assets/gradcam_gallery.png."""
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    from . import viz_style as vs
    from .data import _decode_and_preprocess, load_kaggle_manifest
    from .evaluate import _load_model

    model = _load_model()
    conv_model, head_layers = _build_cam_fns(model)
    thr = json.loads((C.MODELS_DIR / "threshold.json").read_text())["threshold_used"]

    df = load_kaggle_manifest()
    test = df[df["split"] == "test"].reset_index(drop=True)

    # Predict test to categorise into TP/TN/FP/FN.
    probs = model.predict(
        tf.data.Dataset.from_tensor_slices(test["filepath"].to_numpy())
        .map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(C.BATCH_SIZE),
        verbose=0,
    ).ravel()
    test = test.assign(prob=probs, pred=(probs >= thr).astype(int))

    cats = {
        "TP (pneumonia caught)": (test["target"] == 1) & (test["pred"] == 1),
        "TN (normal cleared)":   (test["target"] == 0) & (test["pred"] == 0),
        "FP (false alarm)":      (test["target"] == 0) & (test["pred"] == 1),
        "FN (missed pneumonia)": (test["target"] == 1) & (test["pred"] == 0),
    }

    vs.apply_style()
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(cats), n_per_cat, figsize=(3 * n_per_cat, 3 * len(cats)))
    for r, (label, mask) in enumerate(cats.items()):
        sub = test[mask]
        picks = sub.sample(min(n_per_cat, len(sub)), random_state=seed) if len(sub) else sub
        for c in range(n_per_cat):
            ax = axes[r, c]
            if c >= len(picks):
                ax.axis("off"); continue
            row = picks.iloc[c]
            _, cam = compute_heatmap(conv_model, head_layers, _decode_and_preprocess(row["filepath"]).numpy())
            _overlay(ax, _display_image(row["filepath"]), cam)
            ax.set_title(f"p={row['prob']:.2f}", fontsize=9, color=vs.MUTED)
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center")
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)

    fig.suptitle("Grad-CAM: where the model looks (internal test)", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0, 1, 0.98))
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.ASSETS_DIR / "gradcam_gallery.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved Grad-CAM gallery -> {out}")


if __name__ == "__main__":
    build_gallery()
