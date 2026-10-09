"""Phase 2/3: train Model A (custom CNN) and/or Model B (transfer).

Examples:
    ./venv/bin/python train.py --model both
    ./venv/bin/python train.py --model transfer --head-only --tag frozen
    ./venv/bin/python train.py --model both --subset 64 --epochs-custom 1
        --head-epochs 1 --ft-epochs 1   # smoke test + save/reload check

Per run writes ``{name}_{tag}.keras`` (best), ``{name}_{tag}_final.keras``,
``{name}_{tag}.csv`` and ``{name}_{tag}_meta.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import data_pipeline  # noqa: E402
import models as model_zoo  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for model selection, schedule and ablations."""
    ap = argparse.ArgumentParser(description="Train guava classifiers.")
    ap.add_argument("--model", choices=["custom", "transfer", "both"],
                    default="both")
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE[0])
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--epochs-custom", type=int, default=config.CUSTOM_MAX_EPOCHS)
    ap.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    ap.add_argument("--ft-epochs", type=int, default=config.FT_EPOCHS)
    ap.add_argument("--backbone", default=config.BACKBONE_NAME,
                    choices=["MobileNetV2", "EfficientNetB0", "ResNet50"])
    ap.add_argument("--head-only", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--data-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--tag", default="base")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    return ap.parse_args()


def make_callbacks(stem: str):
    """Checkpoint + early stopping + LR schedule + CSV log."""
    ckpt = config.MODELS_DIR / f"{stem}.keras"
    csv_path = config.LOGS_DIR / f"{stem}.csv"
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.ES_PATIENCE,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.RLROP_FACTOR,
            patience=config.RLROP_PATIENCE, verbose=1),
        tf.keras.callbacks.CSVLogger(csv_path),
    ]
    return cbs, ckpt, csv_path


def check_save_reload(model: tf.keras.Model, sample: tf.Tensor) -> bool:
    """Save/reload round-trip must reproduce predictions."""
    tmp = config.MODELS_DIR / "_smoke_roundtrip.keras"
    before = model.predict(sample, verbose=0)
    model.save(tmp)
    reloaded = tf.keras.models.load_model(tmp, compile=False)
    after = reloaded.predict(sample, verbose=0)
    tmp.unlink(missing_ok=True)
    ok = bool(np.allclose(before, after, atol=1e-6))
    print(f"save/reload prediction parity: {ok}", flush=True)
    return ok


def train_custom(args, train_ds, val_ds, class_names) -> dict:
    """Train Model A from scratch; return JSON-serialisable meta record."""
    stem = f"custom_{args.tag}"
    print(f"=== Model A (custom CNN) [{stem}] ===", flush=True)
    model = model_zoo.build_custom_cnn(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=len(class_names))
    model.summary(print_fn=print, line_length=100)
    cbs, ckpt, _ = make_callbacks(stem)
    cw = (data_pipeline.get_class_weights(args.data_dir, class_names)
          if args.class_weights else None)
    import time
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=args.epochs_custom, callbacks=cbs,
                     class_weight=cw, verbose=2)
    dt = time.time() - t0
    model.save(config.MODELS_DIR / f"{stem}_final.keras")
    sample = next(iter(val_ds.take(1)))[0][:4]
    parity = check_save_reload(
        __import__("keras").models.load_model(ckpt, compile=False), sample)
    total, trainable = model_zoo.count_params(model)
    meta = {"run": stem, "model": "custom_cnn",
            "epochs_run": len(hist.history["loss"]),
            "best_val_accuracy": float(max(hist.history["val_accuracy"])),
            "train_seconds": round(dt, 1), "total_params": total,
            "trainable_params": trainable, "save_reload_parity": parity,
            "seed": args.seed, "img_size": args.img_size}
    (config.LOGS_DIR / f"{stem}_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def train_transfer(args, train_ds, val_ds, class_names) -> dict:
    """Two-phase transfer training (head warm-up, then fine-tune)."""
    stem = f"transfer_{args.backbone}_{args.tag}"
    print(f"=== Model B ({args.backbone}) [{stem}] ===", flush=True)
    model, backbone = model_zoo.build_transfer_model(
        backbone_name=args.backbone,
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=len(class_names))
    model.summary(print_fn=print, line_length=100)
    cbs, ckpt, _ = make_callbacks(stem)
    cw = (data_pipeline.get_class_weights(args.data_dir, class_names)
          if args.class_weights else None)
    import time
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
                   callbacks=cbs, class_weight=cw, verbose=2)
    n1 = len(h1.history["loss"])
    best = float(max(h1.history["val_accuracy"]))
    n2 = 0
    if not args.head_only and args.ft_epochs > 0:
        model = model_zoo.unfreeze_for_fine_tuning(model, backbone)
        total, trainable = model_zoo.count_params(model)
        print(f"Fine-tuning: {trainable}/{total} trainable params", flush=True)
        cbs2, ckpt2, _ = make_callbacks(stem)
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=args.ft_epochs, callbacks=cbs2,
                       class_weight=cw, verbose=2, initial_epoch=n1)
        n2 = len(h2.history["loss"]) - n1
        best = max(best, float(max(h2.history["val_accuracy"])))
        ckpt = ckpt2 if ckpt2.exists() else ckpt
    dt = time.time() - t0
    model.save(config.MODELS_DIR / f"{stem}_final.keras")
    sample = next(iter(val_ds.take(1)))[0][:4]
    parity = check_save_reload(
        __import__("keras").models.load_model(ckpt, compile=False), sample)
    total, trainable = model_zoo.count_params(model)
    meta = {"run": stem, "model": f"transfer_{args.backbone}",
            "head_epochs_run": n1, "ft_epochs_run": n2,
            "best_val_accuracy": best, "train_seconds": round(dt, 1),
            "total_params": total, "trainable_params": trainable,
            "save_reload_parity": parity, "seed": args.seed,
            "img_size": args.img_size, "head_only": args.head_only,
            "augmented": not args.no_augment}
    (config.LOGS_DIR / f"{stem}_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> None:
    """Entry point: build data, train requested model(s), record metadata."""
    args = parse_args()
    data_pipeline.set_seeds(args.seed)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _, class_names = data_pipeline.load_datasets(
        dataset_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size, seed=args.seed,
        use_augmentation=not args.no_augment, subset_train=args.subset)
    print(f"Classes (label order): {class_names}", flush=True)
    metas = []
    if args.model in ("custom", "both"):
        metas.append(train_custom(args, train_ds, val_ds, class_names))
    if args.model in ("transfer", "both"):
        metas.append(train_transfer(args, train_ds, val_ds, class_names))
    print("DONE:", [m["run"] for m in metas], flush=True)


if __name__ == "__main__":
    main()
