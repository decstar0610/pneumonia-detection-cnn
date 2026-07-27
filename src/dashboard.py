"""Phase 5 — Tier-1 evaluation dashboard (PRD §7.5 / BUILD_PLAN Phase 5).

One screenshot-ready image that tells the whole model story at a glance. Six panels, each with a
one-line takeaway caption, all sharing the `viz_style` 2-accent palette:

  1. Internal vs external headline bars (sensitivity / specificity / AUC)
  2. ROC overlay (internal vs external)
  3. Confusion matrices (internal + external)
  4. Calibration reliability (raw vs temperature-scaled)
  5. Subgroup sensitivity (with the 0.85 fairness floor)
  6. Coverage vs accuracy (uncertainty triage)

Reuses persisted artifacts (threshold/temperature/fairness/metrics JSON, external_predictions.csv)
and re-predicts the internal val/test splits for the curve panels.

Run:
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m src.dashboard
"""
from __future__ import annotations

import base64
import json

from . import config as C
from . import viz_style as vs


def _load_everything():
    import pandas as pd

    from .evaluate import _load_model, _predict_split

    model = _load_model()
    yv, pv = _predict_split(model, "val")
    yt, pt = _predict_split(model, "test")
    ext = pd.read_csv(C.MODELS_DIR / "external_predictions.csv")
    d = {
        "yv": yv, "pv": pv, "yt": yt, "pt": pt,
        "ye": ext["y_true"].to_numpy().astype(int), "pe": ext["y_prob"].to_numpy(),
        "thr": json.loads((C.MODELS_DIR / "threshold.json").read_text())["threshold_used"],
        "T": json.loads((C.MODELS_DIR / "temperature.json").read_text())["temperature"],
        "mi": json.loads((C.MODELS_DIR / "metrics_internal.json").read_text()),
        "me": json.loads((C.MODELS_DIR / "metrics_external.json").read_text()),
        "fair": json.loads((C.MODELS_DIR / "fairness.json").read_text()),
    }
    return d


def _panel_bars(ax, d):
    import numpy as np

    keys, labels = ["sensitivity", "specificity", "roc_auc"], ["Sens", "Spec", "AUC"]
    x = np.arange(len(keys)); w = 0.36
    ax.bar(x - w / 2, [d["mi"][k] for k in keys], w, color=vs.PRIMARY, label="internal")
    ax.bar(x + w / 2, [d["me"][k] for k in keys], w, color=vs.ACCENT, label="external")
    ax.axhline(0.92, color=vs.TARGET_LINE, ls=":", lw=1.5)
    for i, k in enumerate(keys):
        ax.text(i - w / 2, d["mi"][k] + 0.01, f"{d['mi'][k]:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, d["me"][k] + 0.01, f"{d['me'][k]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.1); ax.legend(loc="lower right", fontsize=8)
    vs.takeaway(ax, "Does it generalise? Sensitivity holds, specificity drops")


def _panel_roc(ax, d):
    from sklearn.metrics import roc_curve

    for (y, p), color, name in ((("yt", "pt"), vs.PRIMARY, "internal"),
                                (("ye", "pe"), vs.ACCENT, "external")):
        fpr, tpr, _ = roc_curve(d[y], d[p])
        ax.plot(fpr, tpr, color=color, label=f"{name}")
    ax.plot([0, 1], [0, 1], color=vs.MUTED, ls="--", lw=1)
    ax.axhline(0.92, color=vs.TARGET_LINE, ls=":", lw=1.2)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right", fontsize=8)
    vs.takeaway(ax, "ROC — how far does performance shift?")


def _confusion_ax(ax, cm, title):
    import numpy as np

    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                color="white" if v > mat.max() / 2 else vs.MUTED, fontweight="bold")
    ax.set_xticks([0, 1], ["N", "P"], fontsize=8); ax.set_yticks([0, 1], ["N", "P"], fontsize=8)
    ax.set_title(title, fontsize=9, color=vs.MUTED)
    ax.set_xlabel("pred", fontsize=8); ax.set_ylabel("true", fontsize=8)


def _panel_calibration(ax, d):
    from .calibration import _reliability, apply_temperature, expected_calibration_error

    yt, pt, T = d["yt"], d["pt"], d["T"]
    pcal = apply_temperature(pt, T)
    ax.plot([0, 1], [0, 1], color=vs.MUTED, ls="--", lw=1, label="perfect")
    for p, color, name in ((pt, vs.ACCENT, "raw"), (pcal, vs.PRIMARY, "scaled")):
        conf, acc = _reliability(yt, p)
        ece = expected_calibration_error(yt, p)
        ax.plot(conf, acc, marker="o", ms=4, color=color, label=f"{name} ECE={ece:.3f}")
    ax.set_xlabel("mean predicted p"); ax.set_ylabel("empirical rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left", fontsize=8)
    vs.takeaway(ax, "Can I trust its confidence? (calibration)")


def _panel_subgroup(ax, d):
    rows = d["fair"]["sensitivity_by_subgroup"]
    labels = [f"{r['dimension']}: {r['subgroup']}" for r in rows]
    vals = [r["sensitivity"] for r in rows]
    colors = [vs.ACCENT if r["flag"] else vs.PRIMARY for r in rows]
    ax.barh(labels, vals, color=colors)
    ax.axvline(0.85, color=vs.TARGET_LINE, ls=":", lw=1.5)
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.tick_params(axis="y", labelsize=8)
    vs.takeaway(ax, "Who does it fail? (subgroup sensitivity, external)")


def _panel_coverage(ax, d):
    import numpy as np

    from .calibration import apply_temperature

    yt, pt, T, thr = d["yt"], d["pt"], d["T"], d["thr"]
    cal = apply_temperature(pt, T)
    t_cal = float(apply_temperature(np.array([thr]), T)[0])
    correct = ((cal >= t_cal).astype(int) == yt).astype(int)
    conf = np.abs(cal - t_cal)
    order = np.argsort(-conf)
    ks = np.arange(1, len(conf) + 1)
    coverage = ks / len(conf)
    accuracy = np.cumsum(correct[order]) / ks
    ax.plot(coverage, accuracy, color=vs.PRIMARY, lw=2)
    ax.axhline(correct.mean(), color=vs.MUTED, ls="--", lw=1)
    ax.axhline(0.97, color=vs.TARGET_LINE, ls=":", lw=1.5)
    ax.set_xlabel("coverage"); ax.set_ylabel("accuracy"); ax.set_xlim(0, 1)
    vs.takeaway(ax, "How good when it chooses to decide? (triage)")


def build_dashboard() -> None:
    import matplotlib.pyplot as plt

    vs.apply_style()
    d = _load_everything()

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    _panel_bars(fig.add_subplot(gs[0, 0]), d)
    _panel_roc(fig.add_subplot(gs[0, 1]), d)

    # Confusion panel = two small heatmaps in a nested gridspec.
    cell = gs[0, 2].subgridspec(1, 2, wspace=0.5)
    _confusion_ax(fig.add_subplot(cell[0, 0]), d["mi"]["confusion"], "internal")
    _confusion_ax(fig.add_subplot(cell[0, 1]), d["me"]["confusion"], "external")

    _panel_calibration(fig.add_subplot(gs[1, 0]), d)
    _panel_subgroup(fig.add_subplot(gs[1, 1]), d)
    _panel_coverage(fig.add_subplot(gs[1, 2]), d)

    fig.suptitle(
        "PneumoScan — model evaluation dashboard   |   internal test: sens "
        f"{d['mi']['sensitivity']:.2f} / spec {d['mi']['specificity']:.2f} / AUC {d['mi']['roc_auc']:.2f}"
        f"   →   external (RSNA): sens {d['me']['sensitivity']:.2f} / spec {d['me']['specificity']:.2f}"
        f" / AUC {d['me']['roc_auc']:.2f}",
        fontweight="bold", fontsize=13, y=0.98,
    )
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    png = C.ASSETS_DIR / "evaluation_dashboard.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"Saved dashboard PNG -> {png}")
    _write_html(png, d)


def _write_html(png_path, d) -> None:
    """Self-contained HTML: the dashboard image + a metrics table (shareable, no external assets)."""
    b64 = base64.b64encode(png_path.read_bytes()).decode()
    rows = "".join(
        f"<tr><td>{k}</td><td>{d['mi'][k]:.3f}</td><td>{d['me'][k]:.3f}</td></tr>"
        for k in ("sensitivity", "specificity", "precision", "accuracy", "roc_auc", "pr_auc")
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>PneumoScan dashboard</title>
<style>body{{font-family:system-ui,Segoe UI,Arial;margin:24px;color:#222}}
h1{{font-size:20px}} img{{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px}}
table{{border-collapse:collapse;margin-top:16px}} td,th{{border:1px solid #ddd;padding:6px 12px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} caption{{text-align:left;color:#666;margin-bottom:6px}}</style>
<h1>PneumoScan — Tier-1 evaluation dashboard</h1>
<img src="data:image/png;base64,{b64}" alt="evaluation dashboard"/>
<table><caption>Metrics at operating threshold {d['thr']:.3f}</caption>
<tr><th>metric</th><th>internal</th><th>external (RSNA)</th></tr>{rows}</table>
<p style="color:#666;font-size:13px">Research prototype. Not a diagnostic device.</p>"""
    out = C.ASSETS_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved dashboard HTML -> {out}")


if __name__ == "__main__":
    build_dashboard()
