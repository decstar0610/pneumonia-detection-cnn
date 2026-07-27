"""Phase 2 — transfer-learning model (PRD §6.1 / BUILD_PLAN Phase 2).

A DenseNet121 (default) or ResNet50 ImageNet backbone + a custom classification head:

    GlobalAvgPool -> Dense(512,ReLU) -> BN -> Dropout(0.3)
                  -> Dense(256,ReLU) -> BN -> Dropout(0.3)
                  -> Dense(1, sigmoid)

Preprocessing note (important): `src/data.py` already scales to [0,1] and applies
ImageNet mean/std standardisation — this is exactly DenseNet's `preprocess_input`
("torch" mode), so the backbone receives correctly-normalised inputs with NO extra
preprocessing layer. ResNet50's stock weights expect "caffe" mode (BGR, mean-subtract,
no /std), so it is offered only as a comparison backbone — DenseNet121 is the primary.

Two-stage training is driven by `src/train.py`:
  1. `freeze_backbone` -> train the head only (Adam 1e-4).
  2. `unfreeze_top`    -> fine-tune the last conv block (Adam 1e-5), BatchNorm kept frozen.
"""
from __future__ import annotations

from . import config as C

BACKBONES = ("densenet121", "resnet50")


def get_backbone(model):
    """Return the pretrained backbone sub-model (the only nested keras.Model in the graph)."""
    import tensorflow as tf

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No backbone sub-model found in the model graph.")


def build_model(
    backbone_name: str = "densenet121",
    img_size: int = C.IMG_SIZE,
    channels: int = C.CHANNELS,
    dropout: float = 0.3,
):
    """Assemble backbone + custom head. Backbone starts frozen (stage-1 ready)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    backbone_name = backbone_name.lower()
    if backbone_name not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone_name!r}")

    input_shape = (img_size, img_size, channels)
    if backbone_name == "densenet121":
        backbone = tf.keras.applications.DenseNet121(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:  # resnet50
        backbone = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    backbone.trainable = False  # stage 1: feature extractor only

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = backbone(inputs, training=False)  # inference mode -> BN stats stay fixed
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(dropout, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)

    return tf.keras.Model(inputs, outputs, name=f"pneumoscan_{backbone_name}")


def freeze_backbone(model) -> None:
    """Stage 1: freeze the whole backbone; only the custom head trains."""
    get_backbone(model).trainable = False


def unfreeze_top(model, block_prefix: str = "conv5") -> int:
    """Stage 2: unfreeze the top conv block for fine-tuning, keeping BatchNorm frozen.

    Both DenseNet121 and ResNet50 name their final dense/residual block `conv5_*`, so the
    same prefix works for either. BatchNorm layers stay in inference mode — unfreezing them
    on a small medical dataset destabilises the pretrained statistics.

    Returns the number of layers left trainable (for a sanity log).
    """
    from tensorflow.keras.layers import BatchNormalization

    backbone = get_backbone(model)
    backbone.trainable = True
    trainable = 0
    for layer in backbone.layers:
        if isinstance(layer, BatchNormalization) or not layer.name.startswith(block_prefix):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable += 1
    return trainable


def _metrics():
    """Live metrics tracked during training (PRD §6.2): acc / precision / recall / ROC-AUC / PR-AUC.

    `recall` is sensitivity on the positive (PNEUMONIA) class — the primary target (>= 0.92).
    """
    import tensorflow as tf

    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),                 # ROC-AUC (EarlyStopping monitor)
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"),  # PR-AUC (imbalance-aware)
    ]


def compile_model(model, lr: float) -> None:
    """Compile with Adam(lr) + BCE. Class weights are passed at `fit` time, not here."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=_metrics(),
    )
