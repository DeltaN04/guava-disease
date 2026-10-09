"""Phase 4: evaluate trained models on the test set.

For every ``outputs/models/*.keras`` checkpoint (skips ``*_final.keras``
and ``_smoke*`` helpers): loads the best checkpoint, scores the test split,
and writes to ``outputs/results/`` + ``outputs/plots/``:
  * per-model metrics JSON (accuracy, macro P/R/F1, per-class P/R/F1, AUC)
  * confusion matrices (counts + row-normalized), train/val curves overlay
  * ``results_table.csv`` / ``results_table.tex`` (booktabs) incl. params,
    training time and val accuracy from the ``*_meta.json`` logs.

Usage:
    ./venv/bin/python evaluate.py [--data-dir Dataset] [--tag-filter base]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import data_pipeline  # noqa: E402

RUN_LABELS = {
    "custom_base": "A: Custom CNN",
    "transfer_MobileNetV2_base": "B: MobileNetV2 TL",
    "transfer_MobileNetV2_frozen": "B frozen (no FT)",
    "transfer_MobileNetV2_noaug": "B no-augment",
    "custom_leakfree": "A leak-free",
    "transfer_MobileNetV2_leakfree": "B leak-free",
}


def model_runs() -> list[Path]:
    """Best-checkpoint files, excluding finals and smoke-test helpers."""
    return sorted(
        p for p in config.MODELS_DIR.glob("*.keras")
        if not p.name.endswith("_final.keras") and "_smoke" not in p.name
        and not p.name.startswith("_")
    )


def load_test_arrays(data_dir: Path, img_size: int,
                     batch_size: int) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, list[str]]:
    """Return (images, one-hot labels, int labels, class names) for test."""
    _, _, test_ds, class_names = data_pipeline.load_datasets(
        dataset_dir=data_dir, img_size=(img_size, img_size),
        batch_size=batch_size, seed=config.SEED, use_augmentation=False)
    xs, ys = [], []
    for x, y in test_ds:
        xs.append(x.numpy())
        ys.append(y.numpy())
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    return X, Y, np.argmax(Y, axis=1), class_names


def evaluate_model(ckpt: Path, X: np.ndarray, Y: np.ndarray, y: np.ndarray,
                   class_names: list[str]) -> dict:
    """Score one checkpoint; return JSON-serialisable metrics + arrays."""
    model = tf.keras.models.load_model(ckpt, compile=False)
    P = model.predict(X, batch_size=32, verbose=0)
    pred = np.argmax(P, axis=1)
    rep = classification_report(y, pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    try:
        auc_score = float(roc_auc_score(Y, P, multi_class="ovr", average="macro"))
    except ValueError:
        auc_score = float("nan")
    per_class_auc = {}
    for i, c in enumerate(class_names):
        try:
            per_class_auc[c] = float(roc_auc_score(Y[:, i], P[:, i]))
        except ValueError:
            per_class_auc[c] = float("nan")
    return {
        "run": ckpt.stem, "test_accuracy": float(accuracy_score(y, pred)),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "macro_auc": auc_score, "per_class": {
            c: {"precision": float(rep[c]["precision"]),
                "recall": float(rep[c]["recall"]),
                "f1": float(rep[c]["f1-score"]),
                "auc": per_class_auc[c], "support": int(rep[c]["support"])}
            for c in class_names},
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "y_true": y.tolist(), "y_pred": pred.tolist(),
        "y_score": P.tolist(),
    }


def plot_confusion(cm: np.ndarray, labels: list[str], out: Path,
                   title: str, normalize: bool = False) -> None:
    """Confusion matrix heatmap (counts or row-normalized)."""
    disp = (cm.astype(float) / cm.sum(axis=1, keepdims=True)) if normalize else cm
    fmt = ".2f" if normalize else "d"
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(disp, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curves(runs: list[str], out: Path) -> None:
    """Overlay train/val accuracy + loss curves for all runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run in runs:
        csv = config.LOGS_DIR / f"{run}.csv"
        if not csv.exists():
            continue
        h = pd.read_csv(csv)
        label = RUN_LABELS.get(run, run)
        axes[0].plot(h["accuracy"], label=f"{label} train")
        axes[0].plot(h["val_accuracy"], "--", label=f"{label} val")
        axes[1].plot(h["loss"], label=f"{label} train")
        axes[1].plot(h["val_loss"], "--", label=f"{label} val")
    axes[0].set_title("Accuracy")
    axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
    fig.suptitle("Training / validation curves (all runs)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(metrics: dict, class_names: list[str], out: Path) -> None:
    """One-vs-rest ROC curves for the two main models."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for run in ("custom_base", "transfer_MobileNetV2_base"):
        if run not in metrics:
            continue
        y_true = np.array(metrics[run]["y_true"])
        P = np.array(metrics[run]["y_score"])
        for i, c in enumerate(class_names):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), P[:, i])
            ax.plot(fpr, tpr, label=f"{RUN_LABELS.get(run, run)}/{c} "
                                    f"(AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (test set)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def main() -> None:
    """Evaluate all checkpoints; write metrics, figures and result tables."""
    ap = argparse.ArgumentParser(description="Evaluate trained models.")
    ap.add_argument("--data-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE[0])
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--tag-filter", default=None)
    args = ap.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    runs = model_runs()
    if args.tag_filter:
        runs = [p for p in runs if args.tag_filter in p.stem]
    print(f"Checkpoints: {[p.name for p in runs]}", flush=True)
    if not runs:
        print("No checkpoints found; train first.", flush=True)
        return

    X, Y, y, class_names = load_test_arrays(
        Path(args.data_dir), args.img_size, args.batch_size)
    labels = [config.DISPLAY_NAMES.get(c, c) for c in class_names]
    print(f"Test arrays: {X.shape} classes={class_names}", flush=True)

    metrics: dict = {}
    for ckpt in runs:
        print(f"Evaluating {ckpt.name}...", flush=True)
        m = evaluate_model(ckpt, X, Y, y, class_names)
        metrics[m["run"]] = m
        lite = {k: v for k, v in m.items()
                if k not in ("y_true", "y_pred", "y_score")}
        (config.RESULTS_DIR / f"{m['run']}_metrics.json").write_text(
            json.dumps(lite, indent=2))
        cm = np.array(m["confusion_matrix"])
        plot_confusion(cm, labels,
                       config.PLOTS_DIR / f"{m['run']}_cm.png",
                       f"Confusion matrix — {RUN_LABELS.get(m['run'], m['run'])}")
        plot_confusion(cm, labels,
                       config.PLOTS_DIR / f"{m['run']}_cm_norm.png",
                       f"Confusion matrix (row-normalized) — "
                       f"{RUN_LABELS.get(m['run'], m['run'])}",
                       normalize=True)
        print(f"  acc={m['test_accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"AUC={m['macro_auc']:.4f}", flush=True)

    plot_curves(list(metrics), config.PLOTS_DIR / "training_curves.png")
    plot_roc(metrics, class_names, config.PLOTS_DIR / "roc_curves.png")

    rows = []
    for run, m in metrics.items():
        meta_path = config.LOGS_DIR / f"{run}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        rows.append({
            "Model": RUN_LABELS.get(run, run),
            "Test acc": round(m["test_accuracy"], 4),
            "Macro P": round(m["macro_precision"], 4),
            "Macro R": round(m["macro_recall"], 4),
            "Macro F1": round(m["macro_f1"], 4),
            "Macro AUC": (round(m["macro_auc"], 4)
                          if m["macro_auc"] == m["macro_auc"] else "n/a"),
            "Val acc": (round(meta.get("best_val_accuracy", float("nan")), 4)
                        if meta else "n/a"),
            "Params": meta.get("total_params", "n/a"),
            "Trainable": meta.get("trainable_params", "n/a"),
            "Train time": (f"{meta.get('train_seconds', 0) / 60:.1f} min"
                           if meta else "n/a"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "results_table.csv", index=False)
    tex = table.to_latex(index=False, escape=False,
                         column_format="l" + "c" * (len(table.columns) - 1))
    tex = (tex.replace("\\toprule", "\\toprule")
              .replace("\\midrule", "\\midrule")
              .replace("\\bottomrule", "\\bottomrule"))
    (config.RESULTS_DIR / "results_table.tex").write_text(tex)
    print(table.to_string(index=False), flush=True)
    print(f"Wrote {config.RESULTS_DIR} and {config.PLOTS_DIR}", flush=True)


if __name__ == "__main__":
    main()

