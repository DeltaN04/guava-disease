"""Grad-CAM visual explanations with clean axes and normalised heatmaps.

Picks up to ``--per-class`` correctly classified and ``--per-class``
misclassified test images per class, renders the CAM for the **predicted**
class over the original photo, labels the model that produced it, and turns
off every unused axis. Each heatmap is normalised to [0, 1] with an all-zero
guard.

Usage:
    ./venv/bin/python scripts/gradcam.py \
        --ckpt transfer_MobileNetV2_final-s42.keras \
        --data-dir Dataset_leakfree --label "B final (MobileNetV2)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import data_pipeline  # noqa: E402


def last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Deepest Conv2D layer (lives inside the backbone submodel)."""
    for layer in reversed(model.layers):
        inner = [layer] + getattr(layer, "layers", [])
        for cand in reversed(inner):
            if isinstance(cand, tf.keras.layers.Conv2D):
                return cand
    raise ValueError("No Conv2D layer found in model")


def gradcam_heatmap(grad_model: tf.keras.Model, img: np.ndarray,
                    class_idx: int) -> np.ndarray:
    """Grad-CAM heatmap (H, W) normalised to [0, 1], all-zero safe."""
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(np.expand_dims(img, 0))
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return np.zeros(conv_out.shape[1:3], dtype="float32")
    grads = grads[0].numpy()
    weights = grads.mean(axis=(0, 1))
    cam = conv_out[0].numpy() @ weights
    cam = np.maximum(cam, 0)
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-8:
        return np.zeros_like(cam, dtype="float32")
    return ((cam - lo) / (hi - lo)).astype("float32")


def render_panel(img: np.ndarray, cam: np.ndarray, title: str,
                 ax: plt.Axes) -> None:
    """Original | heatmap | overlay side-by-side on one axes row."""
    cam_up = np.array(
        tf.image.resize(cam[..., None], img.shape[:2]).numpy())[..., 0]
    overlay = 0.45 * cam_up[..., None] * np.array([1.0, 0.1, 0.1])
    overlay = np.clip(img + overlay, 0, 1)
    ax.imshow(np.concatenate([img, np.repeat(cam_up[..., None], 3, axis=2),
                              overlay], axis=1))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> None:
    """Generate Grad-CAM panels for correct + misclassified test images."""
    ap = argparse.ArgumentParser(description="Grad-CAM for a checkpoint.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--per-class", type=int, default=3)
    args = ap.parse_args()

    data_pipeline.set_seeds(config.SEED)
    ckpt = config.MODELS_DIR / args.ckpt
    model = tf.keras.models.load_model(ckpt, compile=False)
    label = args.label or ckpt.stem
    prefix = args.out_prefix or f"gradcam_{ckpt.stem}"
    conv = last_conv_layer(model)
    grad_model = tf.keras.Model(model.inputs, [conv.output, model.output])
    print(f"Grad-CAM target layer: {conv.name} {conv.output.shape} "
          f"| model: {label}", flush=True)

    data_dir = Path(args.data_dir)
    _, _, _, class_names = data_pipeline.load_datasets(
        dataset_dir=data_dir, use_augmentation=False)
    labels = [config.DISPLAY_NAMES.get(c, c) for c in class_names]
    files_per_class: dict[str, list[Path]] = {
        c: sorted(p for p in (data_dir / "test" / c).glob("*")
                  if p.is_file() and not p.name.startswith("."))
        for c in class_names
    }

    rng = np.random.default_rng(config.SEED)
    ncols = args.per_class * 2
    fig_all, axes_all = plt.subplots(
        len(class_names), ncols,
        figsize=(3.2 * ncols, 3.2 * len(class_names)), squeeze=False)
    for i, cls in enumerate(class_names):
        files = files_per_class[cls]
        imgs = np.stack([
            np.array(tf.image.resize(
                tf.image.decode_png(tf.io.read_file(str(f)), channels=3),
                [config.IMG_SIZE[0], config.IMG_SIZE[1]])).astype("float32")
            / 255.0 for f in files])
        preds = np.argmax(model.predict(imgs, batch_size=32, verbose=0), axis=1)
        good = [k for k in range(len(files)) if preds[k] == i]
        bad = [k for k in range(len(files)) if preds[k] != i]
        rng.shuffle(good)
        rng.shuffle(bad)
        good = good[:args.per_class]
        bad = bad[:args.per_class]
        print(f"{labels[i]}: {len(good)} correct, {len(bad)} wrong shown",
              flush=True)

        fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.4),
                                 squeeze=False)
        for row, picks in enumerate(([(k, True) for k in good],
                                     [(k, False) for k in bad])):
            for col in range(ncols):
                ax = axes[row][col]
                if col >= len(picks):
                    ax.axis("off")
                    continue
                k, ok = picks[col]
                tag = ("CORRECT" if ok
                       else f"WRONG -> {labels[preds[k]]}")
                cam = gradcam_heatmap(grad_model, imgs[k], int(preds[k]))
                render_panel(
                    imgs[k], cam,
                    f"{labels[i]} | {tag}\n{files[k].name[:26]}", ax)
        for row in range(2):
            for col in range(ncols):
                if col >= len((good if row == 0 else bad)):
                    axes[row][col].axis("off")
        fig.suptitle(f"Grad-CAM — {label} | true class: {labels[i]} "
                     f"(top: correct, bottom: misclassified; predicted-class CAM)")
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / f"{prefix}_{cls}.png", dpi=120)
        plt.close(fig)

        # contact-sheet row: correct examples then wrong examples
        row_axes = axes_all[i]
        picks_all = ([(k, True) for k in good]
                     + [(k, False) for k in bad])[:ncols]
        for col in range(ncols):
            ax = row_axes[col]
            if col >= len(picks_all):
                ax.axis("off")
                continue
            k, ok = picks_all[col]
            tag = "CORRECT" if ok else f"WRONG -> {labels[preds[k]]}"
            cam = gradcam_heatmap(grad_model, imgs[k], int(preds[k]))
            render_panel(imgs[k], cam,
                         f"{labels[i]} | {tag}\n{files[k].name[:22]}", ax)
        for col in range(len(picks_all), ncols):
            row_axes[col].axis("off")
    fig_all.suptitle(f"Grad-CAM contact sheet — {label} "
                     f"({data_dir.name} test; predicted-class CAM)")
    fig_all.tight_layout()
    fig_all.savefig(config.PLOTS_DIR / f"{prefix}_contact_sheet.png", dpi=120)
    plt.close(fig_all)
    print(f"Wrote Grad-CAM panels to {config.PLOTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
