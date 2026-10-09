"""Phase 4: Grad-CAM visual explanations for Model B (transfer).

For each class, picks ~3 correctly classified and ~3 misclassified test
images, renders Grad-CAM heatmaps from the backbone's last convolutional
layer overlaid on the original photo, and saves one panel per class plus
a combined contact sheet to ``outputs/plots/``.

Usage:
    ./venv/bin/python scripts/gradcam.py [--ckpt NAME] [--per-class 3]
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

DEFAULT_CKPT = "transfer_MobileNetV2_base.keras"


def last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Deepest Conv2D layer (lives inside the backbone submodel)."""
    for layer in reversed(model.layers):
        inner = ([layer] + getattr(layer, "layers", []))
        for cand in reversed(inner):
            if isinstance(cand, tf.keras.layers.Conv2D):
                return cand
    raise ValueError("No Conv2D layer found in model")


def gradcam_heatmap(grad_model: tf.keras.Model, img: np.ndarray,
                    class_idx: int) -> np.ndarray:
    """Grad-CAM heatmap (H, W) in [0, 1] for ``class_idx``."""
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(np.expand_dims(img, 0))
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)[0].numpy()
    weights = grads.mean(axis=(0, 1))
    cam = (conv_out[0].numpy() @ weights)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam


def render_panel(img: np.ndarray, cam: np.ndarray, title: str,
                 ax: plt.Axes) -> None:
    """Original | heatmap | overlay side-by-side on one axes row."""
    cam_up = np.array(
        tf.image.resize(cam[..., None], img.shape[:2]).numpy())[..., 0]
    overlay = 0.45 * cam_up[..., None] * np.array([1.0, 0.1, 0.1])
    overlay = np.clip(img + overlay, 0, 1)
    ax.imshow(np.concatenate([img, np.repeat(cam_up[..., None], 3, axis=2),
                              overlay], axis=1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main() -> None:
    """Generate Grad-CAM panels for correct + misclassified test images."""
    ap = argparse.ArgumentParser(description="Grad-CAM for Model B.")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--per-class", type=int, default=3)
    args = ap.parse_args()

    data_pipeline.set_seeds(config.SEED)
    ckpt = config.MODELS_DIR / args.ckpt
    model = tf.keras.models.load_model(ckpt, compile=False)
    conv = last_conv_layer(model)
    grad_model = tf.keras.Model(model.inputs, [conv.output, model.output])
    print(f"Grad-CAM target layer: {conv.name} {conv.output.shape}",
          flush=True)

    _, _, test_ds, class_names = data_pipeline.load_datasets(
        use_augmentation=False)
    paths = [p for p in sorted(
        (config.DATASET_DIR / "test").rglob("*"))
        if p.is_file() and not p.name.startswith(".")]
    # file order of image_dataset_from_directory (no shuffle) is sorted
    per_class_files: dict = {c: [] for c in class_names}
    for cls in class_names:
        per_class_files[cls] = sorted(
            p for p in (config.DATASET_DIR / "test" / cls).glob("*")
            if p.is_file() and not p.name.startswith("."))
    labels = [config.DISPLAY_NAMES.get(c, c) for c in class_names]

    rng = np.random.default_rng(config.SEED)
    fig_all, axes_all = plt.subplots(len(class_names), args.per_class * 2,
                                     figsize=(4 * args.per_class * 2,
                                              4 * len(class_names)))
    for i, cls in enumerate(class_names):
        files = per_class_files[cls]
        imgs = np.stack([
            np.array(tf.image.resize(
                tf.image.decode_png(tf.io.read_file(str(f)), channels=3),
                [config.IMG_SIZE[0], config.IMG_SIZE[1]])).astype("float32")
            / 255.0 for f in files])
        preds = np.argmax(model.predict(imgs, batch_size=32, verbose=0),
                          axis=1)
        good = [k for k in range(len(files)) if preds[k] == i]
        bad = [k for k in range(len(files)) if preds[k] != i]
        rng.shuffle(good)
        rng.shuffle(bad)
        picks = ([(k, True) for k in good[:args.per_class]]
                 + [(k, False) for k in bad[:args.per_class]])
        fig, axes = plt.subplots(2, args.per_class,
                                 figsize=(5 * args.per_class, 8))
        for j, (k, ok) in enumerate(picks):
            row, col = divmod(j, args.per_class)
            tag = "CORRECT" if ok else f"WRONG->as {labels[preds[k]]}"
            cam = gradcam_heatmap(grad_model, imgs[k], int(preds[k]))
            render_panel(imgs[k], cam, f"{files[k].name[:24]}\n{tag}",
                         axes[row][col] if len(picks) > 2 else axes[col])
            if fig_all is not None:
                ax2 = (axes_all[i][j] if len(class_names) > 1 else axes_all[j])
                render_panel(imgs[k], cam,
                             f"{labels[i]}: {files[k].name[:20]}\n{tag}", ax2)
        fig.suptitle(f"Grad-CAM — true class {labels[i]} "
                     f"(row 1: correct, row 2: misclassified)")
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / f"gradcam_{cls}.png", dpi=120)
        plt.close(fig)
    fig_all.suptitle("Grad-CAM contact sheet (Model B, test set)")
    fig_all.tight_layout()
    fig_all.savefig(config.PLOTS_DIR / "gradcam_contact_sheet.png", dpi=120)
    plt.close(fig_all)
    print(f"Wrote Grad-CAM panels to {config.PLOTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
