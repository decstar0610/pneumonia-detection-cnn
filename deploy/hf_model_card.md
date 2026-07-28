---
license: mit
tags:
  - medical-imaging
  - chest-xray
  - pneumonia
  - densenet
  - onnx
  - image-classification
library_name: onnxruntime
pipeline_tag: image-classification
---

# PneumoScan — Chest X-ray Pneumonia Triage

DenseNet121 model that flags likely pneumonia on frontal chest radiographs, built as a
**triage** tool with a trustworthiness-first evaluation (external validation, calibration,
uncertainty-aware abstention, subgroup fairness, Grad-CAM).

> ⚠️ **Not a medical device.** Research/education prototype only. Not for clinical use.

- **Live demo:** https://pneumonia-detection-cnn-chi.vercel.app
- **Code + full write-up:** https://github.com/decstar0610/pneumonia-detection-cnn
- **Model card (limitations):** https://github.com/decstar0610/pneumonia-detection-cnn/blob/main/MODEL_CARD.md

## Files in this repo

| File | What it is |
|---|---|
| `backbone.onnx` | DenseNet121 feature extractor (image 224×224×3 → 7×7×1024), for onnxruntime |
| `gradcam_head.npz` | Classification-head weights (fc1/bn1/fc2/bn2/prob) — run in NumPy for prediction + Grad-CAM |
| `best_model.keras` | Original TensorFlow/Keras model (reference; the ONNX+NumPy path is validated identical) |
| `threshold.json` | Operating threshold (0.324) tuned on validation for sensitivity ≥ 0.92 |
| `temperature.json` | Calibration temperature (T = 0.72) |
| `triage.json` | Calibrated threshold + abstention band for the 3-zone triage |

## Key results

- **Internal test:** sensitivity 0.942, specificity 0.962, ROC-AUC 0.986.
- **External validation (RSNA, adults, n=3000):** sensitivity holds (0.907) but specificity collapses (0.467) — the model was trained on *pediatric* data and false-alarms on unseen adult negatives. This is disclosed, not hidden.
- **PA-view sensitivity 0.615** (vs 0.985 AP) — a real subgroup weakness.

See the [model card](https://github.com/decstar0610/pneumonia-detection-cnn/blob/main/MODEL_CARD.md) for full limitations.

## ⚠️ The resize operator is part of the model

Read this before writing your own preprocessing.

This model was trained and evaluated with **`tf.image.resize`, which is bilinear
*without* antialiasing**. `PIL.Image.resize` — with its default resample *or* with
`BILINEAR` — antialiases when downscaling. It is a different operator, and swapping it
in measurably changes the model.

Measured over the full 586-image held-out test split:

| Resize | Sensitivity | ROC-AUC | False negatives |
|---|---|---|---|
| `PIL.Image.resize(..., BILINEAR)` | 0.9089 | 0.9830 | 39 |
| `tf.image`-equivalent (correct) | **0.9439** | 0.9863 | **24** |

That is **15 lost true positives out of 428** and 24 flipped decisions — from a
substitution that looks harmless and produces per-image probability differences of
~1e-4 on confident cases. The damage is concentrated near the decision threshold, so a
spot check of a few images will not reveal it.

The usage snippet below implements the correct resize.

## Usage (onnxruntime + NumPy head)

```python
import numpy as np, onnxruntime as ort
from PIL import Image

def resize_bilinear(arr, size=224):
    """Matches tf.image.resize(..., antialias=False): half-pixel centers, no antialias."""
    h, w = arr.shape[:2]

    def axis(n_out, n_in):
        src = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
        lo = np.floor(src).astype(int)
        return lo, np.minimum(lo + 1, n_in - 1), (src - lo).astype("float32")

    y0, y1, wy = axis(size, h)
    x0, x1, wx = axis(size, w)
    top = arr[y0][:, x0] + (arr[y0][:, x1] - arr[y0][:, x0]) * wx[None, :, None]
    bot = arr[y1][:, x0] + (arr[y1][:, x1] - arr[y1][:, x0]) * wx[None, :, None]
    return top + (bot - top) * wy[:, None, None]

sess = ort.InferenceSession("backbone.onnx")
H = dict(np.load("gradcam_head.npz"))

img = Image.open("xray.jpg").convert("RGB")
arr = resize_bilinear(np.asarray(img, dtype="float32"))          # NOT img.resize(...)
x = (arr / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
conv = sess.run(None, {sess.get_inputs()[0].name: x[None].astype("float32")})[0][0]

gap = conv.mean((0, 1))
a1 = np.maximum(gap @ H["fc1_W"] + H["fc1_b"], 0)
n1 = H["bn1_gamma"] * (a1 - H["bn1_mean"]) / np.sqrt(H["bn1_var"] + H["bn1_eps"]) + H["bn1_beta"]
a2 = np.maximum(n1 @ H["fc2_W"] + H["fc2_b"], 0)
n2 = H["bn2_gamma"] * (a2 - H["bn2_mean"]) / np.sqrt(H["bn2_var"] + H["bn2_eps"]) + H["bn2_beta"]
prob = 1 / (1 + np.exp(-(n2 @ H["prob_W"][:, 0] + H["prob_b"][0])))
print("P(pneumonia) =", float(prob))
```

Raw sigmoid output is **uncalibrated**. For the full pipeline as deployed, apply
temperature scaling (`temperature.json`, T = 0.72) and then the calibrated operating
threshold and abstention band from `triage.json` —
see [`api/inference.py`](https://github.com/decstar0610/pneumonia-detection-cnn/blob/main/api/inference.py).
