"""Model definitions: custom CNN (Model A) and transfer learning (Model B)."""

from __future__ import annotations

import tensorflow as tf

import config

# ImageNet preprocessing differs by backbone family. Implemented with plain
# ``Rescaling`` / ``Normalization`` layers (no ``Lambda``) so that ``.keras``
# files save and reload without custom objects.
def _preprocess_for(backbone_name: str) -> tf.keras.layers.Layer:
    """Input preprocessing layer matching the backbone's ImageNet training.

    Our pipeline feeds [0, 1] tensors. MobileNetV2 expects [-1, 1].
    EfficientNetB0 contains its own ``Rescaling(1/255)`` + normalization
    layers, so it expects [0, 255] at the model boundary; feeding [-1, 1]
    (the old, wrong behaviour) would make every input negative. ResNet50
    expects caffe-style BGR mean subtraction; we approximate it in RGB order
    (documented limitation, not used for the final model).
    """
    if backbone_name == "MobileNetV2":
        return tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)
    if backbone_name == "EfficientNetB0":
        # EfficientNet's own first layers do 1/255 scaling + normalization.
        return tf.keras.layers.Rescaling(255.0)
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


def make_optimizer(
    name: str = "adam",
    learning_rate: float = 1e-3,
    *,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
    schedule: str = "plateau",
) -> tf.keras.optimizers.Optimizer:
    """Build an optimizer, optionally with a cosine-decay learning-rate schedule.

    ``schedule="plateau"`` keeps a constant LR; the training loop adds a
    ``ReduceLROnPlateau`` callback in that case. ``schedule="cosine"`` uses
    :class:`tf.keras.optimizers.schedules.CosineDecay` over ``epochs`` epochs.
    SGD always uses Nesterov momentum 0.9 (the project's E1 specification).
    AdamW uses Keras' default weight decay (0.004); documented in DECISIONS.
    """
    lr: float | tf.keras.optimizers.schedules.LearningRateSchedule = learning_rate
    if schedule == "cosine":
        if not steps_per_epoch or not epochs:
            raise ValueError("cosine schedule needs steps_per_epoch and epochs")
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=max(1, int(steps_per_epoch) * int(epochs)),
            alpha=0.0,
        )
    elif schedule != "plateau":
        raise ValueError(f"Unknown schedule: {schedule}")

    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=0.9, nesterov=True
        )
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def compile_model(
    model: tf.keras.Model,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    *,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
    schedule: str = "plateau",
) -> tf.keras.Model:
    """Compile with the project's loss/metrics and the requested optimizer."""
    model.compile(
        optimizer=make_optimizer(
            optimizer_name, learning_rate, steps_per_epoch=steps_per_epoch,
            epochs=epochs, schedule=schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model


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
    optimizer_name: str = "adam",
    *,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
    schedule: str = "plateau",
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
    return compile_model(
        model, optimizer_name, learning_rate,
        steps_per_epoch=steps_per_epoch, epochs=epochs, schedule=schedule)


def build_transfer_model(
    backbone_name: str = config.BACKBONE_NAME,
    input_shape: tuple[int, int, int] | None = None,
    num_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.HEAD_LR,
    optimizer_name: str = "adam",
    *,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
    schedule: str = "plateau",
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
    compile_model(
        model, optimizer_name, learning_rate,
        steps_per_epoch=steps_per_epoch, epochs=epochs, schedule=schedule)
    return model, backbone


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    last_n: int = config.FT_UNFREEZE_LAST,
    learning_rate: float = config.FT_LR,
    optimizer_name: str = "adam",
    *,
    steps_per_epoch: int | None = None,
    epochs: int | None = None,
    schedule: str = "plateau",
) -> tf.keras.Model:
    """Unfreeze the last ``last_n`` backbone layers, keep BN frozen, recompile.

    ``last_n <= 0`` leaves the backbone fully frozen (head-only). A negative
    or oversized value means "all backbone layers". BatchNormalization layers
    are always kept frozen during fine-tuning, the recommended practice for
    small datasets.
    """
    if last_n <= 0:
        return model
    backbone.trainable = True
    n_layers = len(backbone.layers)
    n_unfreeze = n_layers if last_n >= n_layers else last_n
    for layer in backbone.layers[:-n_unfreeze]:
        layer.trainable = False
    for layer in backbone.layers[-n_unfreeze:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(
        model, optimizer_name, learning_rate,
        steps_per_epoch=steps_per_epoch, epochs=epochs, schedule=schedule)
    return model


def count_params(model: tf.keras.Model) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    return int(model.count_params()), int(
        sum(tf.size(w).numpy() for w in model.trainable_weights)
    )
