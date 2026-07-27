"""Phase 6 — inference core for the PneumoScan API (PRD §7.2).

Loads the trained model + the persisted operating threshold, temperature, and triage band ONCE,
then turns raw image bytes into the full §7.2 response: calibrated probability, triage zone,
uncertainty flag, and a base64 Grad-CAM overlay. Pure functions here; FastAPI wiring is in main.py.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np

from src import config as C

MODEL_VERSION = "2.0"
_S: dict = {}  # loaded-once state


def warmup() -> None:
    """Load model + calibration/threshold/triage artifacts and JIT the graph on a dummy image."""
    if _S:
        return
    import tensorflow as tf

    from src.gradcam import _build_cam_fns

    model = tf.keras.models.load_model(C.MODELS_DIR / "best_model.keras")
    conv_model, head_layers = _build_cam_fns(model)
    thr = json.loads((C.MODELS_DIR / "threshold.json").read_text())["threshold_used"]
    T = json.loads((C.MODELS_DIR / "temperature.json").read_text())["temperature"]
    triage = json.loads((C.MODELS_DIR / "triage.json").read_text())

    from src.calibration import apply_temperature

    t_cal = float(apply_temperature(np.array([thr]), T)[0])
    _S.update(model=model, conv_model=conv_model, head_layers=head_layers,
              thr=thr, T=T, t_cal=t_cal, band=triage["uncertain_band"],
              mean=np.array(C.IMAGENET_MEAN, "float32"), std=np.array(C.IMAGENET_STD, "float32"))
    # prime the graph so the first real request also meets the <3s budget
    predict_image(_dummy_png())


def _dummy_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (64, 64), color=128).save(buf, "PNG")
    return buf.getvalue()


def _preprocess(image_bytes: bytes):
    """Decode bytes -> (model_input 224x224x3 normalised, display_gray 224x224 in [0,1])."""
    import tensorflow as tf

    raw = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    resized = tf.image.resize(raw, [C.IMG_SIZE, C.IMG_SIZE]) / 255.0
    disp = resized.numpy()
    model_in = ((disp - _S["mean"]) / _S["std"]).astype("float32")
    return model_in, disp.mean(axis=-1)


def _gradcam_b64(model_in, gray) -> str:
    from matplotlib import cm

    from src.gradcam import compute_heatmap

    _, cam = compute_heatmap(_S["conv_model"], _S["head_layers"], model_in)
    heat = cm.get_cmap("jet")(cam)[..., :3]
    blend = 0.6 * np.stack([gray] * 3, axis=-1) + 0.4 * heat
    img8 = (np.clip(blend, 0, 1) * 255).astype("uint8")
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img8).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def predict_image(image_bytes: bytes) -> dict:
    """Full §7.2 contract for one uploaded image. Raises ValueError on undecodable input."""
    if not _S:
        warmup()
    import tensorflow as tf

    try:
        model_in, gray = _preprocess(image_bytes)
    except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError, ValueError) as e:
        raise ValueError(f"Could not decode image: {e}") from e

    raw = float(_S["model"].predict(model_in[None, ...], verbose=0).ravel()[0])
    from src.calibration import apply_temperature

    cal = float(apply_temperature(np.array([raw]), _S["T"])[0])  # calibrated P(pneumonia)
    lo, hi = _S["band"]
    is_uncertain = bool(lo <= cal <= hi)
    is_pneumonia = cal >= _S["t_cal"]

    if is_uncertain:
        zone, rec = "needs_human_review", "Model uncertain — escalate to a radiologist."
    elif is_pneumonia:
        zone, rec = "urgent_review", "Findings suggest pneumonia — flag for radiologist review."
    else:
        zone, rec = "routine", "No urgent finding — routine handling."

    prediction = "Pneumonia" if is_pneumonia else "Normal"
    confidence = cal if is_pneumonia else 1.0 - cal
    return {
        "prediction": prediction,
        "triage_zone": zone,
        "calibrated_confidence": round(confidence, 4),
        "is_uncertain": is_uncertain,
        "probability": {"normal": round(1 - cal, 4), "pneumonia": round(cal, 4)},
        "threshold_used": round(_S["t_cal"], 4),
        "gradcam_overlay": _gradcam_b64(model_in, gray),
        "recommendation": rec,
        "disclaimer": "Research prototype. Not a diagnostic device.",
    }
