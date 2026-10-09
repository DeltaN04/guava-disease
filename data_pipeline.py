"""tf.data input pipelines for the GFDD24 guava dataset.

Conventions (fixed for every run):
  * ``tf.keras.utils.image_dataset_from_directory`` with
    ``label_mode="categorical"`` and ``class_names`` read from disk.
  * Only the training split is shuffled.
  * Pixel rescaling to [0, 1] applies to every split; stochastic
    augmentation layers apply to the training split only.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config


def set_seeds(seed: int = config.SEED) -> None:
    """Fix Python, NumPy and TensorFlow seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def augmentation_layers(seed: int = config.SEED) -> tf.keras.Sequential:
    """Stochastic training-only augmentation block (runs on GPU/CPU in-graph)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.15, seed=seed),
            tf.keras.layers.RandomZoom(0.15, seed=seed),
            tf.keras.layers.RandomContrast(0.1, seed=seed),
            tf.keras.layers.RandomBrightness(0.1, seed=seed),
        ],
        name="train_augmentation",
    )


def load_datasets(
    dataset_dir: Path | str = config.DATASET_DIR,
    img_size: tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    use_augmentation: bool = True,
    subset_train: int | None = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Build (train, val, test) tf.data pipelines.

    Args:
        dataset_dir: Root holding ``train/``, ``val/`` and ``test/`` folders.
        img_size: Target (height, width). Native data is 512x512; we default
            to 224x224 on CPU-only machines (see DECISIONS.md).
        batch_size: Batch size for all splits.
        seed: Shuffle / augmentation seed.
        use_augmentation: If False, train gets rescaling only (ablation b).
        subset_train: If set, keep only the first N training files (smoke tests).

    Returns:
        ``(train_ds, val_ds, test_ds, class_names)`` with one-hot labels.
    """
    dataset_dir = Path(dataset_dir)
    class_names = config.get_class_names(dataset_dir, "train")

    def _split(name: str, shuffle: bool) -> tf.data.Dataset:
        return tf.keras.utils.image_dataset_from_directory(
            dataset_dir / name,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    train_ds = _split("train", shuffle=True)
    val_ds = _split("val", shuffle=False)
    test_ds = _split("test", shuffle=False)

    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)
    if use_augmentation:
        aug = augmentation_layers(seed)

        def _train_map(x, y):
            x = rescale(x)
            x = aug(x, training=True)
            return x, y

        train_ds = train_ds.map(_train_map, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.map(
            lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
    val_ds = val_ds.map(
        lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.map(
        lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    if subset_train is not None:
        n_batches = max(1, subset_train // batch_size)
        train_ds = train_ds.take(n_batches)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


def get_class_weights(
    dataset_dir: Path | str = config.DATASET_DIR,
    class_names: list[str] | None = None,
    split: str = "train",
) -> dict[int, float]:
    """Balanced class weights for the training split (``sklearn``)."""
    dataset_dir = Path(dataset_dir)
    if class_names is None:
        class_names = config.get_class_names(dataset_dir, split)
    y: list[int] = []
    for idx, cls in enumerate(class_names):
        files = sorted((dataset_dir / split / cls).glob("*"))
        files = [f for f in files if f.is_file() and not f.name.startswith(".")]
        y.extend([idx] * len(files))
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(class_names)), y=np.array(y)
    )
    return {int(i): float(w) for i, w in enumerate(weights)}
