"""Model definitions: custom CNN (Model A) and transfer learning (Model B)."""

from __future__ import annotations

import tensorflow as tf

import config

# Backbones whose ImageNet preprocessing is an affine map of [0, 1] inputs to
# [-1, 1]. Implemented with a plain ``Rescaling`` layer (no ``Lambda``) so
# that ``.keras`` files save and reload without custom objects.
_RESACALED_BACKBONES = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
}


def _preprocess_for(backbone_name: str) -> tf.keras.layers.Layer:
    """Input preprocessing layer matching the backbone's ImageNet training."""
    if backbone_name in _RESACALED_BACKBONES:
        # Pipeline inputs are [0, 1]; MobileNetV2 / EfficientNet expect [-1, 1].
        return tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)
    if backbone_name == "ResNet50":
        # Approximation of caffe-style preprocessing in RGB order:
        # [0, 1] -> [0, 255] then per-channel mean subtraction. The exact
        # caffe mode also flips RGB->BGR; we document this approximation.
        return tf.keras.Sequential(
            [
                tf.keras.layers.Rescaling(255.0),
                tf.keras.layers.Normalization(
                    mean=[103.939, 116.779, 123.68], variance=[1.0, 1.0, 1.0]
                ),
            ]
        )
    raise ValueError(f"Unsupported backbone: {backbone_name}")


def _conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    """Conv-ReLU-BN x2 + MaxPool + Dropout(0.25) block for the custom CNN.

    NOTE (bugfix 10 Oct 2026): all BatchNorm layers use ``momentum=0.9``
    instead of the Keras default 0.99. With Adam 1e-3 the from-scratch CNN
    learns very fast (train acc 0.82 after 1 epoch), so batch statistics
    shift rapidly and momentum-0.99 moving averages lag badly: inference
    (moving averages) predicted a single class for every input while
    training-mode (batch stats) hit 0.85 train acc. Momentum 0.9 tracks
    fast-shifting stats on this 2.6k-image regime. Architecture, filter
    counts, dropout rates and LR are unchanged (spec-compliant — the spec
    does not fix BN momentum).
    """
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    return x


def build_custom_cnn(
    input_shape: tuple[int, int, int] | None = None,
    num_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.CUSTOM_LR,
) -> tf.keras.Model:
    """Model A: 4-block CNN (32/64/128/256 filters) trained from scratch."""
    if input_shape is None:
        input_shape = (*config.IMG_SIZE, 3)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for filters in (32, 64, 128, 256):
        x = _conv_block(x, filters)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="custom_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model


def build_transfer_model(
    backbone_name: str = config.BACKBONE_NAME,
    input_shape: tuple[int, int, int] | None = None,
    num_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.HEAD_LR,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Model B: ImageNet backbone (frozen) + classification head.

    Returns ``(model, backbone)``. The backbone is created with
    ``training=False`` semantics (``backbone.trainable = False``) so that
    BatchNorm layers stay in inference mode during head warm-up.
    """
    if input_shape is None:
        input_shape = (*config.IMG_SIZE, 3)
    if backbone_name == "MobileNetV2":
        backbone_fn = tf.keras.applications.MobileNetV2
    elif backbone_name == "EfficientNetB0":
        backbone_fn = tf.keras.applications.EfficientNetB0
    elif backbone_name == "ResNet50":
        backbone_fn = tf.keras.applications.ResNet50
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    inputs = tf.keras.Input(shape=input_shape)
    x = _preprocess_for(backbone_name)(inputs)
    backbone = backbone_fn(
        include_top=False, weights="imagenet", input_tensor=x
    )
    backbone.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"transfer_{backbone_name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model, backbone


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    last_n: int = config.FT_UNFREEZE_LAST,
    learning_rate: float = config.FT_LR,
) -> tf.keras.Model:
    """Unfreeze the last ``last_n`` backbone layers, keep BN frozen, recompile.

    BatchNormalization layers are kept frozen (inference mode) throughout
    fine-tuning, which is the recommended practice for small datasets.
    """
    backbone.trainable = True
    for layer in backbone.layers[:-last_n]:
        layer.trainable = False
    for layer in backbone.layers[-last_n:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model


def count_params(model: tf.keras.Model) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    return int(model.count_params()), int(
        sum(tf.size(w).numpy() for w in model.trainable_weights)
    )
