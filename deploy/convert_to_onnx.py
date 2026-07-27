"""Convert best_model.keras -> ONNX backbone + NumPy head weights, and VALIDATE.

Why: TensorFlow doesn't fit in Render's 512MB free tier. We serve the DenseNet121
backbone via onnxruntime (image -> 7x7x1024) and run the small head (+ Grad-CAM) in
NumPy (see api/onnx_head.py). This script produces the two artifacts the API needs:
  models/backbone.onnx        (the DenseNet121 feature extractor)
  models/gradcam_head.npz     (fc1/bn1/fc2/bn2/prob weights + BN epsilons)
and asserts the ONNX+NumPy pipeline matches the original TF model before we ship.

Run:
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe deploy/convert_to_onnx.py
  ... add --upload to also push both artifacts to the HF Hub model repo.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
ONNX_OUT = MODELS / "backbone.onnx"
HEAD_OUT = MODELS / "gradcam_head.npz"
HF_REPO = "decstzz06/pneumoscan-model"


def export_backbone_and_head():
    import tensorflow as tf

    from api.onnx_head import HEAD_KEYS  # noqa: F401 (ensures module import ok)
    from src.model import get_backbone

    model = tf.keras.models.load_model(MODELS / "best_model.keras")
    backbone = get_backbone(model)
    print(f"Backbone: {backbone.name}  input={backbone.input_shape}  output={backbone.output_shape}")

    # --- head weights -> npz ---
    head = {}
    for name in ("fc1", "fc2", "prob"):
        W, b = model.get_layer(name).get_weights()
        head[f"{name}_W"] = W.astype("float32")
        head[f"{name}_b"] = b.astype("float32")
    for name in ("bn1", "bn2"):
        layer = model.get_layer(name)
        g, beta, mean, var = layer.get_weights()
        head[f"{name}_gamma"] = g.astype("float32")
        head[f"{name}_beta"] = beta.astype("float32")
        head[f"{name}_mean"] = mean.astype("float32")
        head[f"{name}_var"] = var.astype("float32")
        head[f"{name}_eps"] = np.array(layer.epsilon, "float32")
    np.savez(HEAD_OUT, **head)
    print(f"Saved head weights -> {HEAD_OUT}")

    # --- backbone -> SavedModel -> ONNX ---
    with tempfile.TemporaryDirectory() as td:
        saved = Path(td) / "backbone_savedmodel"
        backbone.export(str(saved))  # Keras 3 -> TF SavedModel
        cmd = [sys.executable, "-m", "tf2onnx.convert",
               "--saved-model", str(saved), "--output", str(ONNX_OUT), "--opset", "17"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print(f"Saved ONNX backbone -> {ONNX_OUT}")
    return model  # for validation


def _tf_gradcam7(model, x):
    """TF Grad-CAM at 7x7 (pre-resize), replicating src/gradcam.compute_heatmap's math."""
    import tensorflow as tf

    from src.gradcam import _build_cam_fns

    conv_model, head_layers = _build_cam_fns(model)
    xt = tf.convert_to_tensor(x[None, ...], tf.float32)
    with tf.GradientTape() as tape:
        conv = conv_model(xt, training=False)
        tape.watch(conv)
        h = conv
        for layer in head_layers:
            h = layer(h, training=False)
        score = h[:, 0]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return float(score.numpy()[0]), cam, conv.numpy()


def validate(model, n=8, seed=0):
    import onnxruntime as ort

    from api.onnx_head import head_forward_and_cam, load_head

    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    H = load_head(HEAD_OUT)
    print(f"\nValidating {n} random inputs (ONNX input tensor: '{in_name}') ...")

    rng = np.random.default_rng(seed)
    max_prob_err, min_cam_corr, max_conv_err = 0.0, 1.0, 0.0
    for i in range(n):
        x = rng.standard_normal((224, 224, 3)).astype("float32")  # normalised-range input
        prob_tf, cam_tf, conv_tf = _tf_gradcam7(model, x)
        conv_onnx = sess.run(None, {in_name: x[None, ...]})[0]
        prob_np, cam_np = head_forward_and_cam(conv_onnx, H)

        conv_err = float(np.abs(conv_tf - conv_onnx).max())
        prob_err = abs(prob_tf - prob_np)
        cam_corr = float(np.corrcoef(cam_tf.ravel(), cam_np.ravel())[0, 1])
        max_conv_err = max(max_conv_err, conv_err)
        max_prob_err = max(max_prob_err, prob_err)
        min_cam_corr = min(min_cam_corr, cam_corr)
        print(f"  [{i}] prob TF={prob_tf:.6f} NP={prob_np:.6f} |d|={prob_err:.2e}  "
              f"conv|d|max={conv_err:.2e}  cam corr={cam_corr:.5f}")

    print(f"\nWORST: conv|d|max={max_conv_err:.2e}  prob|d|max={max_prob_err:.2e}  cam corr min={min_cam_corr:.5f}")
    assert max_conv_err < 1e-3, "ONNX backbone diverges from TF"
    assert max_prob_err < 1e-4, "NumPy head prob diverges from TF"
    assert min_cam_corr > 0.99, "NumPy Grad-CAM diverges from TF"
    print("VALIDATION PASSED ✓")


def upload():
    from huggingface_hub import HfApi

    api = HfApi()
    for f in (ONNX_OUT, HEAD_OUT):
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=HF_REPO, repo_type="model")
        print(f"uploaded {f.name} -> {HF_REPO}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="push artifacts to HF Hub after validating")
    args = ap.parse_args()
    model = export_backbone_and_head()
    validate(model)
    if args.upload:
        upload()
    else:
        print("\n(Not uploaded. Re-run with --upload to push to HF Hub.)")


if __name__ == "__main__":
    main()
