"""NumPy reimplementation of the model head (post-backbone) + Grad-CAM.

The DenseNet121 backbone runs in ONNX (image -> 7x7x1024 conv map). The small head
    GAP -> Dense(512,relu) -> BN -> Dense(256,relu) -> BN -> Dense(1,sigmoid)
is tiny, so we run it in NumPy. This gives us the prediction AND — via a hand-coded
backward pass through the same 5 layers — the Grad-CAM weights, with no autograd
framework (TensorFlow) needed at serving time.

`head_forward_and_cam` is the single source of truth shared by the deploy-time
validation (deploy/convert_to_onnx.py) and the live API (api/inference.py). The head
weights live in `gradcam_head.npz` (produced by the convert script).

Grad-CAM equivalence to the TF version (src/gradcam.py): TF takes d(sigmoid_prob)/d(conv)
then averages over the 7x7 spatial dims to get per-channel weights. Since GAP is a spatial
mean, d(prob)/d(conv[i,j,k]) = d(prob)/d(gap[k]) / (H*W), so the spatial-mean weight is
d(prob)/d(gap[k]) up to the constant 1/(H*W) — which cancels in the min-max normalisation.
So `cam = relu(sum_k conv[:,:,k] * d(prob)/d(gap[k]))`, normalised, matches TF Grad-CAM.
"""
from __future__ import annotations

import numpy as np

HEAD_KEYS = (
    "fc1_W", "fc1_b", "bn1_gamma", "bn1_beta", "bn1_mean", "bn1_var", "bn1_eps",
    "fc2_W", "fc2_b", "bn2_gamma", "bn2_beta", "bn2_mean", "bn2_var", "bn2_eps",
    "prob_W", "prob_b",
)


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_head(npz_path) -> dict:
    """Load head weights (as float32 arrays) from the .npz produced at conversion time."""
    data = np.load(npz_path)
    return {k: data[k].astype("float32") for k in HEAD_KEYS}


def head_forward_and_cam(conv, H: dict):
    """Run the head on one backbone output map.

    conv: (7,7,1024) or (1,7,7,1024) float32 — the DenseNet121 final feature map.
    H:    head-weights dict from load_head().
    Returns (raw_prob: float, cam: (7,7) float32 in [0,1]).
    """
    conv = np.asarray(conv, dtype="float32")
    if conv.ndim == 4:
        conv = conv[0]

    # --- forward ---
    gap = conv.mean(axis=(0, 1))                                   # (1024,)
    pre1 = gap @ H["fc1_W"] + H["fc1_b"]; a1 = _relu(pre1)         # (512,)
    inv1 = H["bn1_gamma"] / np.sqrt(H["bn1_var"] + H["bn1_eps"])
    n1 = inv1 * (a1 - H["bn1_mean"]) + H["bn1_beta"]
    pre2 = n1 @ H["fc2_W"] + H["fc2_b"]; a2 = _relu(pre2)          # (256,)
    inv2 = H["bn2_gamma"] / np.sqrt(H["bn2_var"] + H["bn2_eps"])
    n2 = inv2 * (a2 - H["bn2_mean"]) + H["bn2_beta"]
    logit = float(n2 @ H["prob_W"][:, 0] + H["prob_b"][0])
    prob = float(_sigmoid(logit))

    # --- backward: d(prob)/d(gap) ---
    dlogit = prob * (1.0 - prob)                                   # d sigmoid / d logit
    dn2 = dlogit * H["prob_W"][:, 0]                               # (256,)
    dpre2 = (dn2 * inv2) * (pre2 > 0)                              # through BN2 then ReLU
    dn1 = dpre2 @ H["fc2_W"].T                                     # (512,)
    dpre1 = (dn1 * inv1) * (pre1 > 0)                              # through BN1 then ReLU
    dgap = dpre1 @ H["fc1_W"].T                                    # (1024,)

    # --- Grad-CAM map ---
    cam = _relu((conv * dgap).sum(axis=-1))                        # (7,7)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return prob, cam.astype("float32")
