"""Benchmark backbone cost: params, minutes/epoch, inference ms/image.

Loads the E4 checkpoints (MobileNetV2 / EfficientNetB0 / ResNet50) and times
single-image and batch-32 inference on the leak-free test set. Writes
``outputs/results/backbone_bench.csv`` and
``outputs/plots/backbone_cost.png``.

Usage:
    ./venv/bin/python scripts/bench_backbones.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import data_pipeline  # noqa: E402

DEFAULT_RUNS = ["transfer_MobileNetV2_e1-adam",
                "transfer_EfficientNetB0_e4-effnet",
                "transfer_ResNet50_e4-resnet"]


def time_inference(model: tf.keras.Model, X: np.ndarray,
                   batch: int, n: int = 40) -> float:
    """Milliseconds per image at a given batch size (after warm-up)."""
    f = tf.function(lambda z: model(z, training=False))
    z = tf.convert_to_tensor(X[:batch])
    f(z).numpy()
    n_batches = max(1, n // batch)
    t0 = time.perf_counter()
    for i in range(n_batches):
        f(tf.convert_to_tensor(X[i * batch:(i + 1) * batch])).numpy()
    dt = time.perf_counter() - t0
    return dt / (n_batches * batch) * 1000.0


def main() -> None:
    """Benchmark the three E4 backbones; save CSV and cost-vs-accuracy plot."""
    data_pipeline.set_seeds(config.SEED)
    _, _, test_ds, class_names = data_pipeline.load_datasets(
        dataset_dir=config.LEAKFREE_DATASET_DIR, use_augmentation=False)
    xs = []
    for x, _ in test_ds:
        xs.append(x.numpy())
        if sum(len(a) for a in xs) >= 64:
            break
    X = np.concatenate(xs)[:64]

    rows = []
    for run in DEFAULT_RUNS:
        ckpt = config.MODELS_DIR / f"{run}.keras"
        if not ckpt.exists():
            print(f"SKIP {run}: checkpoint missing")
            continue
        meta_path = config.LOGS_DIR / f"{run}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        model = tf.keras.models.load_model(ckpt, compile=False)
        b1 = time_inference(model, X, batch=1, n=20)
        b32 = time_inference(model, X, batch=32, n=64)
        epochs = max(1, meta.get("epochs_run", 1))
        rows.append({
            "run": run,
            "backbone": meta.get("backbone", run),
            "params": meta.get("total_params", int(model.count_params())),
            "best_val_acc": meta.get("best_val_accuracy", float("nan")),
            "train_minutes": meta.get("train_seconds", 0) / 60,
            "minutes_per_epoch": meta.get("train_seconds", 0) / 60 / epochs,
            "infer_ms_batch1": b1,
            "infer_ms_batch32": b32,
        })
        print(f"{run}: {b1:.1f} ms/img (b1), {b32:.1f} ms/img (b32)")

    if not rows:
        print("No E4 checkpoints found; train E4 first.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "backbone_bench.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(df["backbone"], df["best_val_acc"], color="#2E75B6")
    axes[0].set_ylim(0.8, 1.01)
    axes[0].set_ylabel("Best validation accuracy")
    axes[0].set_title("Accuracy (leak-free validation)")
    axes[1].bar(df["backbone"], df["infer_ms_batch1"], color="#1B3B6F")
    axes[1].set_ylabel("Inference ms / image (batch 1)")
    axes[1].set_title("CPU inference cost")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "backbone_cost.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {config.RESULTS_DIR / 'backbone_bench.csv'} and "
          f"{config.PLOTS_DIR / 'backbone_cost.png'}")


if __name__ == "__main__":
    main()
