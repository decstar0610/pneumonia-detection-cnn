"""Phase 6 — inference core for the PneumoScan API (PRD §7.2).

Serving runtime is TensorFlow-free so it fits Render's 512MB free tier: the
DenseNet121 backbone runs in onnxruntime (image -> 7x7x1024) and the small head +
Grad-CAM run in NumPy (see api/onnx_head.py). Loads the backbone + head + persisted
threshold/temperature/triage-band ONCE, then turns raw image bytes into the full
§7.2 response: calibrated probability, triage zone, uncertainty flag, base64 Grad-CAM
overlay. Numerically validated to match the original TF model (deploy/convert_to_onnx.py).
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np

from src import config as C

MODEL_VERSION = "2.0"
_S: dict = {}  # loaded-once state


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _apply_temperature(p, T):
    """Calibrated prob = sigmoid(logit(p)/T). Pure NumPy (matches src.calibration)."""
    return _sigmoid(_logit(p) / T)


def warmup() -> None:
    """Load ONNX backbone + head weights + threshold/temperature/triage, then prime once."""
    if _S:
        return
    import onnxruntime as ort

    from api.onnx_head import load_head

    sess = ort.InferenceSession(str(C.MODELS_DIR / "backbone.onnx"),
                                providers=["CPUExecutionProvider"])
    head = load_head(C.MODELS_DIR / "gradcam_head.npz")
    thr = json.loads((C.MODELS_DIR / "threshold.json").read_text())["threshold_used"]
    T = json.loads((C.MODELS_DIR / "temperature.json").read_text())["temperature"]
    triage = json.loads((C.MODELS_DIR / "triage.json").read_text())

    t_cal = float(_apply_temperature(np.array([thr]), T)[0])
    _S.update(sess=sess, in_name=sess.get_inputs()[0].name, head=head,
              thr=thr, T=T, t_cal=t_cal, band=triage["uncertain_band"],
              mean=np.array(C.IMAGENET_MEAN, "float32"), std=np.array(C.IMAGENET_STD, "float32"))
    predict_image(_dummy_png())  # prime the graph so the first real request is warm


def _dummy_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (64, 64), color=128).save(buf, "PNG")
    return buf.getvalue()


def _preprocess(image_bytes: bytes):
    """Decode bytes -> (model_input 224x224x3 normalised, display_gray 224x224 in [0,1])."""
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Could not decode image: {e}") from e
    img = img.resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR)
    disp = (np.asarray(img, dtype="float32") / 255.0)          # (224,224,3) in [0,1]
    model_in = ((disp - _S["mean"]) / _S["std"]).astype("float32")
    return model_in, disp.mean(axis=-1)


def _jet(x):
    """Minimal 'jet' colormap: x (H,W) in [0,1] -> (H,W,3) RGB in [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _gradcam_b64(cam7, gray) -> str:
    """cam7 (7x7 [0,1]) -> jet overlay on the grayscale X-ray -> base64 PNG data URI."""
    from PIL import Image

    cam = np.asarray(Image.fromarray((cam7 * 255).astype("uint8"))
                     .resize((C.IMG_SIZE, C.IMG_SIZE), Image.BILINEAR), dtype="float32") / 255.0
    heat = _jet(cam)
    blend = 0.6 * np.stack([gray] * 3, axis=-1) + 0.4 * heat
    img8 = (np.clip(blend, 0, 1) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(img8).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def predict_image(image_bytes: bytes) -> dict:
    """Full §7.2 contract for one uploaded image. Raises ValueError on undecodable input."""
    if not _S:
        warmup()
    from api.onnx_head import head_forward_and_cam

    model_in, gray = _preprocess(image_bytes)
    conv = _S["sess"].run(None, {_S["in_name"]: model_in[None, ...]})[0]
    raw, cam7 = head_forward_and_cam(conv, _S["head"])   # raw P(pneumonia), 7x7 Grad-CAM

    cal = float(_apply_temperature(np.array([raw]), _S["T"])[0])  # calibrated P(pneumonia)
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
        "gradcam_overlay": _gradcam_b64(cam7, gray),
        "recommendation": rec,
        "disclaimer": "Research prototype. Not a diagnostic device.",
    }
