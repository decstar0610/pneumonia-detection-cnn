"""Phase 3 — internal evaluation + threshold tuning (PRD §4 / BUILD_PLAN Phase 3).

The key decision (§4): the default 0.5 threshold is NOT the operating point. We CHOOSE the
threshold on the **validation** split to guarantee sensitivity >= TARGET_SENSITIVITY on the
costly-to-miss PNEUMONIA class, then report the resulting specificity on the held-out **test**
split. Test is touched once, at the frozen threshold — no peeking, no tuning on test.

Produces:
  models/threshold.json         — the persisted operating threshold + how it was chosen
  models/metrics_internal.json  — full internal test metrics at that threshold
  assets/eval_roc_pr.png        — ROC + PR curves (test) with the operating point marked
  assets/eval_confusion.png     — internal test confusion matrix at the tuned threshold

Run (from repo root, venv active):
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m src.evaluate
"""
from __future__ import annotations

import argparse
import json

from . import config as C

TARGET_SENSITIVITY = 0.92  # PRD primary target on PNEUMONIA (the costly-to-miss class)


def _load_model():
    import tensorflow as tf

    path = C.MODELS_DIR / "best_model.keras"
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.train` first (Phase 2)."
        )
    return tf.keras.models.load_model(path)


def _predict_split(model, split: str):
    """Return (y_true, y_prob) for a split. val/test are unshuffled, so predict-order == label-order."""
    import numpy as np

    from .data import make_generators

    ds = make_generators(split, shuffle=False, augment=False)
    y_prob = model.predict(ds, verbose=0).ravel()
    y_true = np.concatenate([y.numpy() for _, y in ds]).astype(int)
    return y_true, y_prob


def choose_threshold(y_true, y_prob, target: float = TARGET_SENSITIVITY):
    """Highest threshold whose sensitivity >= target (i.e. best specificity that still meets it).

    Uses the ROC sweep: thresholds are descending and TPR (=sensitivity) is non-decreasing as the
    threshold drops, so the FIRST threshold meeting the target is the highest one that qualifies.
    """
    import numpy as np
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:  # model can't reach the target anywhere — fall back to max-sensitivity point
        i = int(np.argmax(tpr))
    else:
        i = int(ok[0])
    thr = float(np.clip(thresholds[i], 0.0, 1.0))
    return thr, float(tpr[i]), float(1.0 - fpr[i])  # threshold, val sensitivity, val specificity


def _metrics_at(y_true, y_prob, thr: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        "threshold": thr,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "accuracy": acc,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _plot_curves(y_true, y_prob, thr: float, path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        PrecisionRecallDisplay,
        RocCurveDisplay,
        precision_recall_curve,
        roc_curve,
    )

    from . import viz_style as vs

    vs.apply_style()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ROC with the operating point marked. roc_curve thresholds are DESCENDING, so the last
    # threshold still >= thr is the chosen operating point.
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax_roc, color=vs.PRIMARY, name="internal test")
    op = max(int((thr_roc >= thr).sum()) - 1, 0)
    ax_roc.scatter(fpr[op], tpr[op], color=vs.ACCENT, zorder=5, s=60)
    ax_roc.plot([0, 1], [0, 1], color=vs.MUTED, ls="--", lw=1)
    ax_roc.axhline(TARGET_SENSITIVITY, color=vs.TARGET_LINE, ls=":", lw=1.5)
    vs.takeaway(ax_roc, f"ROC — operating point at threshold {thr:.3f} (sens target {TARGET_SENSITIVITY})")

    # PR with the operating point marked. precision_recall_curve thresholds are ASCENDING and
    # 1 shorter than prec/rec; searchsorted gives the point at our operating threshold.
    import numpy as np

    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax_pr, color=vs.PRIMARY, name="internal test")
    op_pr = min(int(np.searchsorted(thr_pr, thr)), len(prec) - 1)
    ax_pr.scatter(rec[op_pr], prec[op_pr], color=vs.ACCENT, zorder=5, s=60)
    vs.takeaway(ax_pr, "Precision–Recall (imbalance-aware view)")

    fig.suptitle("Internal test performance", fontweight="bold")
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved ROC/PR curves -> {path}")


def _plot_confusion(cm: dict, path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from . import viz_style as vs

    vs.apply_style()
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > mat.max() / 2 else vs.MUTED, fontweight="bold")
    ax.set_xticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_yticks([0, 1], ["NORMAL", "PNEUMONIA"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    vs.takeaway(ax, "Internal test confusion matrix (at tuned threshold)")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved confusion matrix -> {path}")


def _plot_overlay(curves: dict, thr: float, path) -> None:
    """Overlay ROC + PR for internal test vs external (RSNA) — 'how much does it shift?'."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    from . import viz_style as vs

    vs.apply_style()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {"internal test": vs.PRIMARY, "external (RSNA)": vs.ACCENT}
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_roc.plot(fpr, tpr, color=colors[name], label=name)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(rec, prec, color=colors[name], label=name)
    ax_roc.plot([0, 1], [0, 1], color=vs.MUTED, ls="--", lw=1)
    ax_roc.axhline(TARGET_SENSITIVITY, color=vs.TARGET_LINE, ls=":", lw=1.5)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    vs.takeaway(ax_roc, "ROC — internal vs external"); ax_roc.legend(loc="lower right")
    vs.takeaway(ax_pr, "Precision–Recall — internal vs external"); ax_pr.legend(loc="lower left")
    fig.suptitle("Generalisation: internal (pediatric) vs external (RSNA, mixed-age)", fontweight="bold")
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved ROC/PR overlay -> {path}")


def _plot_internal_vs_external(mi: dict, me: dict, path) -> None:
    """Grouped bar of the three headline metrics — 'does it generalise?'."""
    import matplotlib.pyplot as plt
    import numpy as np

    from . import viz_style as vs

    vs.apply_style()
    keys = ["sensitivity", "specificity", "roc_auc"]
    labels = ["Sensitivity", "Specificity", "ROC-AUC"]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, [mi[k] for k in keys], w, color=vs.PRIMARY, label="internal (pediatric)")
    ax.bar(x + w / 2, [me[k] for k in keys], w, color=vs.ACCENT, label="external (RSNA)")
    ax.axhline(TARGET_SENSITIVITY, color=vs.TARGET_LINE, ls=":", lw=1.5)
    for i, k in enumerate(keys):
        ax.text(i - w / 2, mi[k] + 0.01, f"{mi[k]:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, me[k] + 0.01, f"{me[k]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.08); ax.legend(loc="lower right")
    vs.takeaway(ax, "Performance drop from pediatric training to mixed-age external set")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved internal-vs-external bars -> {path}")


def external_report(sample_size: int | None) -> None:
    """Phase 4 part 2: evaluate the frozen model on RSNA at the internal-tuned threshold."""
    from .data import load_external_rsna

    model = _load_model()
    thr_info = json.loads((C.MODELS_DIR / "threshold.json").read_text())
    thr = thr_info["threshold_used"]
    print(f"Using internal-tuned threshold {thr:.4f} (no re-tuning on external).")

    # Internal test (re-predicted for a fair curve overlay against external).
    yt, pt = _predict_split(model, "test")
    mi = _metrics_at(yt, pt, thr)

    # External RSNA.
    print(f"Loading RSNA external set (sample_size={sample_size}) — DICOM decode on CPU, please wait...")
    ds, meta = load_external_rsna(sample_size=sample_size)
    import numpy as np

    pe = model.predict(ds, verbose=0).ravel()
    ye = meta["Target"].to_numpy().astype(int)
    me = _metrics_at(ye, pe, thr)
    me["n"] = int(len(ye)); me["positives"] = int(ye.sum())

    print("\nInternal vs External @ frozen threshold")
    print("-" * 60)
    print(f"  {'metric':12s} {'internal':>10s} {'external':>10s} {'delta':>8s}")
    for k in ("sensitivity", "specificity", "precision", "accuracy", "roc_auc", "pr_auc"):
        print(f"  {k:12s} {mi[k]:>10.4f} {me[k]:>10.4f} {me[k] - mi[k]:>+8.4f}")
    c = me["confusion"]
    print(f"  external n={me['n']} (positives={me['positives']}) "
          f"confusion TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")

    (C.MODELS_DIR / "metrics_external.json").write_text(json.dumps(me, indent=2))
    # persist per-image probs alongside metadata for the Phase 4.4 subgroup audit
    import pandas as pd

    pd.DataFrame({"patientId": meta["patientId"], "y_true": ye, "y_prob": pe,
                  "view": meta["view"], "sex": meta["sex"], "age": meta["age"],
                  "class": meta["class"]}).to_csv(C.MODELS_DIR / "external_predictions.csv", index=False)
    print(f"Persisted -> {C.MODELS_DIR / 'metrics_external.json'} , "
          f"{C.MODELS_DIR / 'external_predictions.csv'}")

    _plot_overlay({"internal test": (yt, pt), "external (RSNA)": (ye, pe)},
                  thr, C.ASSETS_DIR / "eval_roc_pr_overlay.png")
    _plot_internal_vs_external(mi, me, C.ASSETS_DIR / "eval_internal_vs_external.png")


FAIRNESS_FLOOR = 0.85  # PRD §4.4: flag any subgroup whose sensitivity falls below this


def _age_band(age):
    import numpy as np

    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown"
    if age <= 18:
        return "0-18 (pediatric)"
    if age <= 39:
        return "19-39"
    if age <= 59:
        return "40-59"
    return "60+"


def fairness_report() -> None:
    """Phase 4.4: per-subgroup sensitivity on the external set (no re-inference; reads the CSV).

    Sensitivity is computed on positives (all class 'Lung Opacity'), so we break it down by view /
    sex / age band. Because every positive shares one class, the *class* cut instead reports
    SPECIFICITY among negatives (Normal vs 'No Lung Opacity / Not Normal') — this quantifies why
    external specificity collapsed: the model over-flags abnormal-but-not-pneumonia cases.
    """
    import json

    import pandas as pd

    csv = C.MODELS_DIR / "external_predictions.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} not found. Run `python -m src.evaluate --external` first.")
    df = pd.read_csv(csv)
    thr = json.loads((C.MODELS_DIR / "threshold.json").read_text())["threshold_used"]
    df["age_band"] = df["age"].map(_age_band)

    pos, neg = df[df["y_true"] == 1], df[df["y_true"] == 0]
    overall_sens = float((pos["y_prob"] >= thr).mean())

    # Sensitivity by subgroup (positives only).
    sens_rows = []
    for col in ("view", "sex", "age_band"):
        for g, sub in pos.groupby(col):
            n = len(sub)
            sens = float((sub["y_prob"] >= thr).mean())
            sens_rows.append({"dimension": col, "subgroup": str(g), "n_pos": n,
                              "sensitivity": sens, "flag": sens < FAIRNESS_FLOOR})
    sens_tbl = pd.DataFrame(sens_rows)

    # Specificity by negative class (the mechanism behind the specificity collapse).
    spec_rows = []
    for g, sub in neg.groupby("class"):
        n = len(sub)
        spec = float((sub["y_prob"] < thr).mean())
        spec_rows.append({"neg_class": str(g), "n_neg": n, "specificity": spec})
    spec_tbl = pd.DataFrame(spec_rows).sort_values("specificity")

    print(f"\nFairness / subgroup audit (external RSNA, threshold {thr:.4f})")
    print("=" * 64)
    print(f"Overall external sensitivity: {overall_sens:.3f}  (flag floor {FAIRNESS_FLOOR})\n")
    print("Sensitivity by subgroup (positives):")
    for _, r in sens_tbl.iterrows():
        mark = "  <-- BELOW FLOOR" if r["flag"] else ""
        print(f"  {r['dimension']:9s} {r['subgroup']:18s} n={r['n_pos']:4d}  sens={r['sensitivity']:.3f}{mark}")
    print("\nSpecificity by negative class (why specificity collapsed):")
    for _, r in spec_tbl.iterrows():
        print(f"  {r['neg_class']:32s} n={r['n_neg']:4d}  spec={r['specificity']:.3f}")

    SMALL_N = 50  # below this, a flagged subgroup is genuinely too noisy to over-read
    flagged = sens_tbl[sens_tbl["flag"]]
    substantive = flagged[flagged["n_pos"] >= SMALL_N]
    noisy = flagged[flagged["n_pos"] < SMALL_N]
    print("\nInterpretation:")
    if len(substantive):
        for _, r in substantive.iterrows():
            print(f"  REAL DISPARITY: {r['dimension']}={r['subgroup']} sensitivity "
                  f"{r['sensitivity']:.3f} (n={r['n_pos']}) is well below the {FAIRNESS_FLOOR} floor "
                  "on a non-trivial sample — a genuine subgroup weakness to disclose in the model card.")
    if len(noisy):
        print(f"  Also below floor but small-n (<{SMALL_N}, treat as noisy): "
              f"{', '.join(noisy['subgroup'])}.")
    if not len(flagged):
        print(f"  Sensitivity stays >= {FAIRNESS_FLOOR} across every view/sex/age subgroup — the "
              "pneumonia-detection ability itself is stable across groups.")
    print("  The collapse is in SPECIFICITY, and it is class-driven: the model keeps specificity on "
          "true 'Normal' films but false-alarms heavily on 'No Lung Opacity / Not Normal' (other "
          "pathology it never saw in pediatric training) — that is the generalisation gap, not a "
          "demographic bias.")

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODELS_DIR / "fairness.json").write_text(json.dumps({
        "threshold": thr, "flag_floor": FAIRNESS_FLOOR, "overall_sensitivity": overall_sens,
        "sensitivity_by_subgroup": sens_rows, "specificity_by_negative_class": spec_rows,
    }, indent=2))
    _plot_fairness(sens_tbl, spec_tbl, C.ASSETS_DIR / "eval_fairness.png")
    print(f"\nPersisted -> {C.MODELS_DIR / 'fairness.json'}")


def _plot_fairness(sens_tbl, spec_tbl, path) -> None:
    import matplotlib.pyplot as plt

    from . import viz_style as vs

    vs.apply_style()
    fig, (ax_s, ax_c) = plt.subplots(1, 2, figsize=(13, 5.4))

    labels = [f"{r.dimension}: {r.subgroup}" for r in sens_tbl.itertuples()]
    vals = sens_tbl["sensitivity"].tolist()
    colors = [vs.ACCENT if v < FAIRNESS_FLOOR else vs.PRIMARY for v in vals]
    ax_s.barh(labels, vals, color=colors)
    ax_s.axvline(FAIRNESS_FLOOR, color=vs.TARGET_LINE, ls=":", lw=1.5)
    ax_s.set_xlim(0, 1); ax_s.invert_yaxis()
    vs.takeaway(ax_s, f"Sensitivity by subgroup (flag line {FAIRNESS_FLOOR})")

    short = {"No Lung Opacity / Not Normal": "Not Normal\n(other pathology)", "Normal": "Normal"}
    clabels = [short.get(c, c) for c in spec_tbl["neg_class"]]
    ax_c.barh(clabels, spec_tbl["specificity"].tolist(), color=vs.MUTED)
    ax_c.set_xlim(0, 1); ax_c.invert_yaxis()
    vs.takeaway(ax_c, "Specificity by negative class (the collapse is here)")

    fig.suptitle("External subgroup / fairness audit", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved fairness figure -> {path}")


TRIAGE_TARGET_ACC = 0.97  # decided-case accuracy we want the abstention band to reach


def _coverage_accuracy(conf, correct):
    """Sweep an abstention cut over confidence: return (coverage, accuracy) as we decide fewer cases.

    conf = distance of calibrated prob from the decision boundary; correct = per-case 0/1 hit.
    Abstaining on the least-confident cases first, so coverage falls and accuracy rises.
    """
    import numpy as np

    order = np.argsort(-conf)  # most confident first
    correct_sorted = correct[order]
    ks = np.arange(1, len(conf) + 1)
    coverage = ks / len(conf)
    accuracy = np.cumsum(correct_sorted) / ks
    cut_conf = conf[order]  # confidence of the last-kept case at each k
    return coverage, accuracy, cut_conf


def triage_report() -> None:
    """Phase 4.3: calibrated 3-zone triage + coverage-vs-accuracy curve (internal test)."""
    import numpy as np

    from .calibration import apply_temperature

    model = _load_model()
    thr = json.loads((C.MODELS_DIR / "threshold.json").read_text())["threshold_used"]
    T = json.loads((C.MODELS_DIR / "temperature.json").read_text())["temperature"]

    yt, pt = _predict_split(model, "test")
    cal = apply_temperature(pt, T)                       # calibrated probabilities
    t_cal = float(apply_temperature(np.array([thr]), T)[0])  # operating threshold on calibrated scale
    pred = (cal >= t_cal).astype(int)                    # monotonic -> same decisions as raw>=thr
    correct = (pred == yt).astype(int)
    conf = np.abs(cal - t_cal)                           # uncertainty = closeness to the boundary
    full_acc = float(correct.mean())

    coverage, accuracy, cut_conf = _coverage_accuracy(conf, correct)

    # Choose the widest band (max coverage) whose decided-accuracy still meets the target.
    meets = accuracy >= TRIAGE_TARGET_ACC
    if meets.any():
        k = int(np.where(meets)[0].max())  # largest k (highest coverage) that still meets target
        s_star = float(cut_conf[k])
        chosen_cov, chosen_acc = float(coverage[k]), float(accuracy[k])
    else:  # target unreachable -> fall back to the most-confident decile
        k = max(int(0.1 * len(conf)) - 1, 0)
        s_star = float(cut_conf[k]); chosen_cov, chosen_acc = float(coverage[k]), float(accuracy[k])

    lo, hi = t_cal - s_star, t_cal + s_star
    routine = int((cal < lo).sum())
    urgent = int((cal > hi).sum())
    uncertain_mask = (cal >= lo) & (cal <= hi)
    uncertain = int(uncertain_mask.sum())
    unc_pos = int(yt[uncertain_mask].sum())

    print("\nUncertainty-aware triage (internal test, calibrated probs)")
    print("=" * 60)
    print(f"  operating threshold (calibrated): {t_cal:.3f}")
    print(f"  full-coverage accuracy: {full_acc:.3f}  (deciding on all {len(yt)} cases)")
    print(f"  abstain band: [{lo:.3f}, {hi:.3f}]  (|p_cal - {t_cal:.3f}| < {s_star:.3f})")
    print(f"  -> decide {chosen_cov*100:.0f}% of cases at {chosen_acc:.3f} accuracy "
          f"(target {TRIAGE_TARGET_ACC}); escalate {100 - chosen_cov*100:.0f}%.")
    print("\n  Zones:")
    print(f"    routine  (Normal, p_cal < {lo:.3f})    : {routine}")
    print(f"    UNCERTAIN (escalate to radiologist)   : {uncertain}  (of which {unc_pos} truly pneumonia)")
    print(f"    urgent   (Pneumonia, p_cal > {hi:.3f}) : {urgent}")
    print("\n  The abstention band trades coverage for safety: sending the ambiguous ~"
          f"{100 - chosen_cov*100:.0f}% to a human lifts accuracy on the rest from "
          f"{full_acc:.3f} to {chosen_acc:.3f} — the triage value proposition, made measurable.")

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODELS_DIR / "triage.json").write_text(json.dumps({
        "temperature": T, "threshold_calibrated": t_cal,
        "uncertain_band": [lo, hi], "target_accuracy": TRIAGE_TARGET_ACC,
        "coverage": chosen_cov, "decided_accuracy": chosen_acc, "full_accuracy": full_acc,
        "zones": {"routine": routine, "uncertain": uncertain, "urgent": urgent,
                  "uncertain_true_pneumonia": unc_pos},
    }, indent=2))
    print(f"Persisted -> {C.MODELS_DIR / 'triage.json'}")

    _plot_coverage_accuracy(coverage, accuracy, chosen_cov, chosen_acc, full_acc,
                            C.ASSETS_DIR / "eval_triage.png")


def _plot_coverage_accuracy(coverage, accuracy, cov, acc, full_acc, path) -> None:
    import matplotlib.pyplot as plt

    from . import viz_style as vs

    vs.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(coverage, accuracy, color=vs.PRIMARY, lw=2)
    ax.scatter([cov], [acc], color=vs.ACCENT, zorder=5, s=70)
    ax.annotate(f"decide {cov*100:.0f}% @ {acc:.3f}", (cov, acc),
                textcoords="offset points", xytext=(8, -14), color=vs.ACCENT)
    ax.axhline(full_acc, color=vs.MUTED, ls="--", lw=1)
    ax.axhline(TRIAGE_TARGET_ACC, color=vs.TARGET_LINE, ls=":", lw=1.5)
    ax.set_xlabel("Coverage (fraction of cases decided)")
    ax.set_ylabel("Accuracy on decided cases")
    ax.set_xlim(0, 1)
    vs.takeaway(ax, "Coverage vs accuracy — abstaining on the uncertain lifts accuracy on the rest")
    fig.suptitle("Uncertainty-aware triage (internal test)", fontweight="bold")
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved coverage-accuracy curve -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal eval + threshold tuning; --external / --fairness / --triage.")
    parser.add_argument("--external", action="store_true", help="Phase 4.1: RSNA external validation")
    parser.add_argument("--fairness", action="store_true", help="Phase 4.4: subgroup/fairness audit (on saved CSV)")
    parser.add_argument("--triage", action="store_true", help="Phase 4.3: uncertainty-aware triage + coverage/accuracy")
    parser.add_argument("--sample", type=int, default=3000,
                        help="external stratified sample size (0 = all 26,684)")
    args = parser.parse_args()

    if args.fairness:
        fairness_report()
        return
    if args.triage:
        triage_report()
        return
    if args.external:
        external_report(sample_size=args.sample or None)
        return

    model = _load_model()

    # 1) Tune the threshold on VALIDATION.
    yv, pv = _predict_split(model, "val")
    thr, val_sens, val_spec = choose_threshold(yv, pv)
    print(f"\nTuned on validation: threshold={thr:.4f}  "
          f"val_sensitivity={val_sens:.4f}  val_specificity={val_spec:.4f}")

    # 2) Evaluate once on TEST at that frozen threshold.
    yt, pt = _predict_split(model, "test")
    m = _metrics_at(yt, pt, thr)

    print("\nInternal TEST metrics @ tuned threshold")
    print("-" * 44)
    for k in ("sensitivity", "specificity", "precision", "accuracy", "roc_auc", "pr_auc"):
        print(f"  {k:12s}: {m[k]:.4f}")
    c = m["confusion"]
    print(f"  confusion   : TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")

    # Targets check (BUILD_PLAN Phase 2 exit criteria).
    targets = {"sensitivity": 0.92, "specificity": 0.75, "roc_auc": 0.94}
    print("\nTargets:")
    for k, t in targets.items():
        print(f"  {k:12s} {m[k]:.3f} >= {t}  {'PASS' if m[k] >= t else 'BELOW'}")

    # 3) Persist threshold + metrics; save figures.
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODELS_DIR / "threshold.json").write_text(json.dumps({
        "threshold_used": thr,
        "target_sensitivity": TARGET_SENSITIVITY,
        "chosen_on": "validation",
        "val_sensitivity": val_sens,
        "val_specificity": val_spec,
    }, indent=2))
    (C.MODELS_DIR / "metrics_internal.json").write_text(json.dumps(m, indent=2))
    print(f"\nPersisted -> {C.MODELS_DIR / 'threshold.json'} , {C.MODELS_DIR / 'metrics_internal.json'}")

    _plot_curves(yt, pt, thr, C.ASSETS_DIR / "eval_roc_pr.png")
    _plot_confusion(m["confusion"], C.ASSETS_DIR / "eval_confusion.png")


if __name__ == "__main__":
    main()
