"""Phase 2 — two-stage transfer-learning training (PRD §6.2 / BUILD_PLAN Phase 2).

Stage 1: freeze backbone, train the head (Adam 1e-4).
Stage 2: unfreeze the top conv block, fine-tune (Adam 1e-5), BatchNorm kept frozen.

Loss: BCE with balanced class weights (~3:1 imbalance). Callbacks: EarlyStopping on
`val_auc`, ReduceLROnPlateau, ModelCheckpoint(best by val_auc). A single best checkpoint
persists across BOTH stages, so the saved model is the global best.

Outputs:
  models/best_model.keras   — best model by val_auc (reloadable in evaluate.py / serving)
  assets/training_history.png — loss / AUC / recall(sensitivity) across both stages

Run (from repo root, venv active). CPU-only is fine but slow — smoke-test first:
    ./.venv/Scripts/python.exe -m src.train --smoke            # ~2 batches, verifies wiring
    ./.venv/Scripts/python.exe -m src.train                    # full run (DenseNet121)
    ./.venv/Scripts/python.exe -m src.train --backbone resnet50
"""
from __future__ import annotations

import argparse

from . import config as C
from . import model as M


def _plot_history(histories: list, path) -> None:
    """Concatenate stage histories and plot loss / val_auc / val_recall across both stages."""
    import matplotlib.pyplot as plt

    from . import viz_style as vs

    vs.apply_style()

    def series(key: str) -> list:
        out: list = []
        for h in histories:
            out.extend(h.history.get(key, []))
        return out

    stage1_len = len(histories[0].history.get("loss", [])) if histories else 0
    epochs = range(1, len(series("loss")) + 1)

    panels = [
        ("loss", "val_loss", "BCE loss"),
        ("auc", "val_auc", "ROC-AUC"),
        ("recall", "val_recall", "Recall = sensitivity (target 0.92)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (tr_key, val_key, title) in zip(axes, panels):
        ax.plot(epochs, series(tr_key), color=vs.PRIMARY, label="train")
        ax.plot(epochs, series(val_key), color=vs.ACCENT, label="val")
        if stage1_len and stage1_len < len(list(epochs)):
            ax.axvline(stage1_len + 0.5, color=vs.MUTED, ls="--", lw=1)
        if val_key == "val_recall":
            ax.axhline(0.92, color=vs.TARGET_LINE, ls=":", lw=1.5)
        ax.set_xlabel("epoch")
        vs.takeaway(ax, title)
        ax.legend(loc="best")
    fig.suptitle(
        "Training history (dashed = freeze→fine-tune handoff)", fontweight="bold"
    )
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved training history -> {path}")


def _callbacks(ckpt_path):
    import tensorflow as tf

    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt_path), monitor="val_auc", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.3, patience=3,
            min_lr=1e-7, verbose=1,
        ),
    ]


def train(
    backbone: str = "densenet121",
    head_epochs: int = 15,
    finetune_epochs: int = 20,
    smoke: bool = False,
) -> None:
    import tensorflow as tf

    from .data import compute_class_weights, make_generators

    tf.keras.utils.set_random_seed(C.SEED)

    train_ds = make_generators("train")
    val_ds = make_generators("val")
    class_weight = compute_class_weights()
    print(f"Class weights (balanced): {class_weight}")

    fit_kwargs = dict(validation_data=val_ds, class_weight=class_weight, verbose=1)
    if smoke:
        # Tiny run just to prove the pipeline trains end-to-end (a few batches, 1 epoch/stage).
        head_epochs = finetune_epochs = 1
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(1)
        fit_kwargs["validation_data"] = val_ds
        print("SMOKE mode: 2 train batches / 1 val batch, 1 epoch per stage.")

    ckpt_path = C.MODELS_DIR / "best_model.keras"
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = _callbacks(ckpt_path)

    model = M.build_model(backbone)
    model.summary(print_fn=lambda s: print(s) if smoke else None)

    # --- Stage 1: frozen backbone, train the head -----------------------------
    print("\n=== Stage 1: train head (backbone frozen, Adam 1e-4) ===")
    M.freeze_backbone(model)
    M.compile_model(model, lr=1e-4)
    h1 = model.fit(train_ds, epochs=head_epochs, callbacks=callbacks, **fit_kwargs)

    # --- Stage 2: unfreeze top block, fine-tune -------------------------------
    n_trainable = M.unfreeze_top(model)
    print(f"\n=== Stage 2: fine-tune top block ({n_trainable} layers, Adam 1e-5) ===")
    M.compile_model(model, lr=1e-5)  # recompile after changing trainable flags
    h2 = model.fit(
        train_ds,
        epochs=head_epochs + finetune_epochs,
        initial_epoch=len(h1.history["loss"]),
        callbacks=callbacks,
        **fit_kwargs,
    )

    _plot_history([h1, h2], C.ASSETS_DIR / "training_history.png")

    # Report the best model (restored by EarlyStopping / reloaded from checkpoint).
    if ckpt_path.exists():
        best = tf.keras.models.load_model(ckpt_path)
        print("\nBest model (by val_auc) — validation metrics:")
        results = best.evaluate(val_ds, verbose=0, return_dict=True)
        for k, v in results.items():
            print(f"  {k:10s}: {v:.4f}")
        print(f"\nSaved best model -> {ckpt_path}")
    else:
        print("WARNING: no checkpoint written (val_auc never improved?).")


def main() -> None:
    p = argparse.ArgumentParser(description="Two-stage transfer-learning trainer (PneumoScan).")
    p.add_argument("--backbone", choices=M.BACKBONES, default="densenet121")
    p.add_argument("--head-epochs", type=int, default=15, help="max stage-1 epochs")
    p.add_argument("--finetune-epochs", type=int, default=20, help="max stage-2 epochs")
    p.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    args = p.parse_args()
    train(
        backbone=args.backbone,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
