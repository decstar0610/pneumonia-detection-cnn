"""Phase 7 — export the evaluation story as data for the Tier-2 web dashboard (PRD §7.5).

The static Tier-1 dashboard bakes its numbers into a PNG; the interactive Model
Report tab needs the same numbers as JSON. This script recomputes them from the
persisted artifacts and writes ONE file the frontend imports, so nothing in the UI
is a hand-typed constant.

Sources
  * internal test set  — re-scored here through the exact serving path
    (ONNX backbone + NumPy head, api/onnx_head.py), so the dashboard agrees with
    what /predict returns rather than with a separate TensorFlow pass.
  * external RSNA set  — read from models/external_predictions.csv (per-image
    probabilities persisted by `evaluate.py --external`); no re-inference needed.
  * threshold / temperature / triage / fairness — the persisted JSONs.

Everything downstream of the model is expressed on the CALIBRATED probability
scale (p_cal = sigmoid(logit(p)/T)), which is the scale the API compares against
`threshold_used`. Temperature scaling is monotonic, so ROC/PR are unchanged and
the operating point maps 0.3245 (raw) -> 0.2653 (calibrated).

Run:
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m src.export_report
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .calibration import _reliability, apply_temperature, expected_calibration_error

MODEL_VERSION = "2.0"
SWEEP_POINTS = 201  # threshold grid for the live slider: 0.000 … 1.000 in 0.005 steps
MAX_CURVE_POINTS = 320  # per ROC/PR curve, after uniform subsampling
MAX_COVERAGE_POINTS = 240


# --- helpers ----------------------------------------------------------------


def _round(value: float, digits: int = 5) -> float:
    return float(np.round(float(value), digits))


def _subsample(*arrays, limit: int):
    """Uniformly thin parallel arrays to `limit` points, always keeping both ends."""
    n = len(arrays[0])
    if n <= limit:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, limit).round().astype(int))
    return [np.asarray(a)[idx] for a in arrays]


def _counts_at(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, int]:
    pred = prob >= threshold
    return {
        "tp": int(np.sum(pred & (y_true == 1))),
        "fp": int(np.sum(pred & (y_true == 0))),
        "tn": int(np.sum(~pred & (y_true == 0))),
        "fn": int(np.sum(~pred & (y_true == 1))),
    }


def _rates(c: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    total = tp + fp + tn + fn
    return {
        "sensitivity": _round(tp / (tp + fn) if tp + fn else 0.0),
        "specificity": _round(tn / (tn + fp) if tn + fp else 0.0),
        "precision": _round(tp / (tp + fp) if tp + fp else 0.0),
        "accuracy": _round((tp + tn) / total if total else 0.0),
    }


def _sweep(y_true: np.ndarray, prob: np.ndarray) -> list[dict]:
    """Every operating point the threshold slider can land on — real counts, not a model."""
    grid = np.linspace(0.0, 1.0, SWEEP_POINTS)
    points = []
    for t in grid:
        counts = _counts_at(y_true, prob, float(t))
        points.append({"t": _round(t, 3), **counts, **_rates(counts)})
    return points


def _roc_pr(y_true: np.ndarray, prob: np.ndarray) -> tuple[list[dict], list[dict], float, float]:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = float(auc(fpr, tpr))
    f, t = _subsample(fpr, tpr, limit=MAX_CURVE_POINTS)
    roc = [{"fpr": _round(a, 4), "tpr": _round(b, 4)} for a, b in zip(f, t)]

    precision, recall, _ = precision_recall_curve(y_true, prob)
    pr_auc = float(auc(recall, precision))
    r, p = _subsample(recall, precision, limit=MAX_CURVE_POINTS)
    pr = [{"recall": _round(a, 4), "precision": _round(b, 4)} for a, b in zip(r, p)]

    return roc, pr, roc_auc, pr_auc


# --- data loading -----------------------------------------------------------


def _score_internal_test() -> tuple[np.ndarray, np.ndarray]:
    """Re-score the held-out internal test split through the deployed serving path."""
    import onnxruntime as ort

    from api.onnx_head import head_forward_and_cam, load_head

    rows = []
    with open(C.SPLIT_MANIFEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "test":
                rows.append((Path(row["filepath"]), int(row["target"])))
    if not rows:
        raise SystemExit("No test rows in the split manifest — run data/resplit.py first.")

    session = ort.InferenceSession(
        str(C.MODELS_DIR / "backbone.onnx"), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    head = load_head(C.MODELS_DIR / "gradcam_head.npz")
    mean = np.array(C.IMAGENET_MEAN, "float32")
    std = np.array(C.IMAGENET_STD, "float32")

    from PIL import Image

    from api.inference import _resize_bilinear  # same operator the API serves with

    y_true = np.zeros(len(rows), dtype=int)
    prob = np.zeros(len(rows), dtype=float)
    print(f"Scoring {len(rows)} internal test studies through the ONNX serving path…")
    for i, (path, target) in enumerate(rows):
        img = Image.open(path).convert("RGB")
        resized = _resize_bilinear(np.asarray(img, dtype="float32"), C.IMG_SIZE)
        x = ((resized / 255.0 - mean) / std).astype("float32")
        conv = session.run(None, {input_name: x[None, ...]})[0]
        raw, _ = head_forward_and_cam(conv, head)
        y_true[i] = target
        prob[i] = raw
        if (i + 1) % 150 == 0:
            print(f"  {i + 1}/{len(rows)}")
    return y_true, prob


def _load_external() -> tuple[np.ndarray, np.ndarray]:
    path = C.MODELS_DIR / "external_predictions.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path} — run `python -m src.evaluate --external` first.")
    y_true, prob = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            y_true.append(int(row["y_true"]))
            prob.append(float(row["y_prob"]))
    return np.array(y_true), np.array(prob)


# --- sections ---------------------------------------------------------------


def _dataset_block(y_true: np.ndarray, prob_cal: np.ndarray, t_cal: float) -> dict:
    roc, pr, roc_auc, pr_auc = _roc_pr(y_true, prob_cal)
    counts = _counts_at(y_true, prob_cal, t_cal)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": _round(float(y_true.mean())),
        "roc_auc": _round(roc_auc),
        "pr_auc": _round(pr_auc),
        "confusion": counts,
        **_rates(counts),
        "roc": roc,
        "pr": pr,
        "sweep": _sweep(y_true, prob_cal),
    }


def _calibration_block(y_true: np.ndarray, raw: np.ndarray, cal: np.ndarray, T: float) -> dict:
    def bins(p):
        conf, acc = _reliability(y_true, p)
        return [
            {"confidence": _round(c, 4), "empirical": _round(a, 4)}
            for c, a in zip(conf, acc)
            if not (np.isnan(c) or np.isnan(a))
        ]

    return {
        "temperature": T,
        "ece_raw": _round(expected_calibration_error(y_true, raw), 4),
        "ece_scaled": _round(expected_calibration_error(y_true, cal), 4),
        "bins_raw": bins(raw),
        "bins_scaled": bins(cal),
        "n_bins": 10,
    }


def _triage_block(y_true: np.ndarray, cal: np.ndarray, t_cal: float, persisted: dict) -> dict:
    """Coverage-vs-accuracy: accuracy on the cases the model chose to decide."""
    correct = ((cal >= t_cal).astype(int) == y_true).astype(int)
    conf = np.abs(cal - t_cal)
    order = np.argsort(-conf)
    ks = np.arange(1, len(conf) + 1)
    coverage = ks / len(conf)
    accuracy = np.cumsum(correct[order]) / ks
    cov, acc = _subsample(coverage, accuracy, limit=MAX_COVERAGE_POINTS)
    return {
        "band": [_round(v, 4) for v in persisted["uncertain_band"]],
        "threshold_calibrated": _round(persisted["threshold_calibrated"], 4),
        "target_accuracy": persisted["target_accuracy"],
        "coverage": _round(persisted["coverage"], 4),
        "decided_accuracy": _round(persisted["decided_accuracy"], 4),
        "full_accuracy": _round(persisted["full_accuracy"], 4),
        "zones": persisted["zones"],
        "curve": [{"coverage": _round(c, 4), "accuracy": _round(a, 4)} for c, a in zip(cov, acc)],
    }


def main() -> None:
    threshold = json.loads((C.MODELS_DIR / "threshold.json").read_text())
    temperature = json.loads((C.MODELS_DIR / "temperature.json").read_text())
    triage = json.loads((C.MODELS_DIR / "triage.json").read_text())
    fairness = json.loads((C.MODELS_DIR / "fairness.json").read_text())

    T = float(temperature["temperature"])
    thr_raw = float(threshold["threshold_used"])
    t_cal = float(apply_temperature(np.array([thr_raw]), T)[0])

    y_int, p_int_raw = _score_internal_test()
    p_int_cal = apply_temperature(p_int_raw, T)
    y_ext, p_ext_raw = _load_external()
    p_ext_cal = apply_temperature(p_ext_raw, T)

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model_version": MODEL_VERSION,
        "probability_scale": "calibrated",
        "threshold": {
            "raw": _round(thr_raw, 4),
            "calibrated": _round(t_cal, 4),
            "target_sensitivity": threshold["target_sensitivity"],
            "chosen_on": threshold["chosen_on"],
            "val_sensitivity": _round(threshold["val_sensitivity"]),
            "val_specificity": _round(threshold["val_specificity"]),
        },
        "temperature": {
            "value": T,
            "fit_on": temperature["fit_on"],
            "selection": temperature["selection"],
        },
        "datasets": {
            "internal": {
                "label": "Internal test",
                "source": "Kaggle chest X-ray (Kermany/Mooney) — pediatric, same source as training",
                "note": "Held-out 10% of the stratified re-split. Never seen during training.",
            },
            "external": {
                "label": "External (RSNA)",
                "source": "RSNA Pneumonia Detection subset (NIH-derived) — adult, different source",
                "note": "Stratified sample scored once at the frozen threshold. No retraining, no re-tuning.",
            },
        },
        "internal": _dataset_block(y_int, p_int_cal, t_cal),
        "external": _dataset_block(y_ext, p_ext_cal, t_cal),
        "calibration": _calibration_block(y_int, p_int_raw, p_int_cal, T),
        "triage": _triage_block(y_int, p_int_cal, t_cal, triage),
        "fairness": {
            "flag_floor": fairness["flag_floor"],
            "overall_sensitivity": _round(fairness["overall_sensitivity"]),
            "by_subgroup": [
                {
                    "dimension": row["dimension"],
                    "subgroup": row["subgroup"],
                    "n_pos": row["n_pos"],
                    "sensitivity": _round(row["sensitivity"]),
                    "flag": row["flag"],
                }
                for row in fairness["sensitivity_by_subgroup"]
            ],
            "by_negative_class": [
                {
                    "neg_class": row["neg_class"],
                    "n_neg": row["n_neg"],
                    "specificity": _round(row["specificity"]),
                }
                for row in fairness["specificity_by_negative_class"]
            ],
        },
    }

    payload = json.dumps(report, indent=1)
    targets = [
        C.ROOT / "frontend" / "src" / "data" / "model_report.json",
        C.ASSETS_DIR / "model_report.json",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        print(f"Wrote {target}  ({len(payload) / 1024:.0f} KB)")

    ib, eb = report["internal"], report["external"]
    print("\nSanity check (frozen calibrated threshold "
          f"{report['threshold']['calibrated']}):")
    print(f"  internal  sens {ib['sensitivity']:.4f}  spec {ib['specificity']:.4f}  auc {ib['roc_auc']:.4f}")
    print(f"  external  sens {eb['sensitivity']:.4f}  spec {eb['specificity']:.4f}  auc {eb['roc_auc']:.4f}")
    print("  (compare against models/metrics_internal.json and metrics_external.json)")


if __name__ == "__main__":
    main()
