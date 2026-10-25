"""Train Model A (custom CNN) and/or Model B (transfer) with resumable runs.

Examples:
    # fixed Model A: select on val_accuracy, at least 15 epochs
    ./venv/bin/python train.py --model custom --tag es-fixed --epochs-custom 40 \
        --es-monitor val_accuracy --start-from-epoch 15 --task T1

    # one optimisation-grid run (val only; no test evaluation)
    ./venv/bin/python train.py --model transfer --data-dir Dataset_leakfree \
        --tag e1_sgd --head-epochs 8 --ft-epochs 8 --optimizer sgd \
        --head-lr 1e-2 --ft-lr 1e-3 --task E1

    # resume an interrupted custom run from its best checkpoint
    ./venv/bin/python train.py --model custom --data-dir Dataset_leakfree \
        --tag es-fixed-leakfree --resume outputs/models/custom_leakfree.keras \
        --resume-epoch 16 --resume-best 0.9737 --resume-seconds 5600 \
        --epochs-custom 40 --es-monitor val_accuracy --task T1

Every run writes ``{name}_{tag}.keras`` (best), ``{name}_{tag}_final.keras``,
``{name}_{tag}.csv`` and ``{name}_{tag}_meta.json``. If the meta file and best
checkpoint already exist the run is skipped unless ``--force`` is passed.
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
    """Parse CLI flags for model selection, schedule, optimiser and ablation."""
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

    # optimisation controls (D11+)
    ap.add_argument("--optimizer", default="adam",
                    choices=["adam", "adamw", "sgd", "rmsprop"])
    ap.add_argument("--custom-lr", type=float, default=config.CUSTOM_LR)
    ap.add_argument("--head-lr", type=float, default=config.HEAD_LR)
    ap.add_argument("--ft-lr", type=float, default=config.FT_LR)
    ap.add_argument("--unfreeze-last", type=int, default=config.FT_UNFREEZE_LAST,
                    help="0=head only, negative/oversized=all layers")
    ap.add_argument("--lr-schedule", default="plateau",
                    choices=["plateau", "cosine"])
    ap.add_argument("--es-monitor", default="val_accuracy",
                    choices=["val_accuracy", "val_loss"])
    ap.add_argument("--es-patience", type=int, default=config.ES_PATIENCE)
    ap.add_argument("--start-from-epoch", type=int, default=0,
                    help="EarlyStopping warm-up (absolute epoch number)")

    # bookkeeping / resumability
    ap.add_argument("--task", default="misc", help="experiment id, e.g. E1")
    ap.add_argument("--note", default="", help="free-text note stored in meta")
    ap.add_argument("--force", action="store_true", help="overwrite existing run")
    ap.add_argument("--resume", default=None,
                    help="custom checkpoint to resume from (custom model only)")
    ap.add_argument("--resume-epoch", type=int, default=0)
    ap.add_argument("--resume-best", type=float, default=0.0)
    ap.add_argument("--resume-seconds", type=float, default=0.0)
    return ap.parse_args()


def steps_per_epoch(ds: tf.data.Dataset) -> int:
    """Number of batches per epoch (known for directory datasets)."""
    n = int(tf.data.experimental.cardinality(ds).numpy())
    if n <= 0:
        raise ValueError("Dataset cardinality is unknown; cannot schedule LR")
    return n


def make_callbacks(stem: str, *, append: bool = False,
                   monitor: str = "val_accuracy", patience: int = 8,
                   start_from_epoch: int = 0,
                   initial_best: float | None = None,
                   schedule: str = "plateau"):
    """Checkpoint + early stopping + optional plateau LR + CSV log."""
    ckpt = config.MODELS_DIR / f"{stem}.keras"
    csv_path = config.LOGS_DIR / f"{stem}.csv"
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor=monitor, save_best_only=True, verbose=1,
            initial_value_threshold=(initial_best if monitor == "val_accuracy"
                                     else None)),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True,
            verbose=1, start_from_epoch=start_from_epoch),
        tf.keras.callbacks.CSVLogger(csv_path, append=append),
    ]
    if schedule == "plateau":
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.RLROP_FACTOR,
            patience=config.RLROP_PATIENCE, min_lr=1e-7, verbose=1))
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


def _best_epoch(history: dict, initial_epoch: int) -> int | None:
    """1-based absolute epoch with the highest val_accuracy (or val_loss)."""
    if "val_accuracy" in history and history["val_accuracy"]:
        return initial_epoch + int(np.argmax(history["val_accuracy"])) + 1
    if "val_loss" in history and history["val_loss"]:
        return initial_epoch + int(np.argmin(history["val_loss"])) + 1
    return None


def _run_exists(stem: str) -> dict | None:
    """Return existing meta if both meta and best checkpoint exist."""
    meta_path = config.LOGS_DIR / f"{stem}_meta.json"
    ckpt = config.MODELS_DIR / f"{stem}.keras"
    if meta_path.exists() and ckpt.exists():
        meta = json.loads(meta_path.read_text())
        print(f"SKIP {stem}: meta+checkpoint already exist "
              f"(best val {meta.get('best_val_accuracy')})", flush=True)
        return meta
    return None


def train_custom(args, train_ds, val_ds, class_names, spb: int) -> dict:
    """Train Model A from scratch (or resume a checkpoint); return meta."""
    stem = f"custom_{args.tag}"
    existing = None if args.force else _run_exists(stem)
    if existing:
        return existing
    print(f"=== Model A (custom CNN) [{stem}] ===", flush=True)

    initial_epoch = 0
    prior_seconds = 0.0
    prior_best = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        print(f"Resuming from {resume_path} at epoch {args.resume_epoch}",
              flush=True)
        model = tf.keras.models.load_model(resume_path, compile=False)
        initial_epoch = int(args.resume_epoch)
        prior_seconds = float(args.resume_seconds)
        prior_best = float(args.resume_best)
        model = model_zoo.compile_model(
            model, args.optimizer, args.custom_lr,
            steps_per_epoch=spb,
            epochs=max(1, args.epochs_custom - initial_epoch),
            schedule=args.lr_schedule)
    else:
        model = model_zoo.build_custom_cnn(
            input_shape=(args.img_size, args.img_size, 3),
            num_classes=len(class_names),
            learning_rate=args.custom_lr,
            optimizer_name=args.optimizer,
            steps_per_epoch=spb, epochs=args.epochs_custom,
            schedule=args.lr_schedule)
    model.summary(print_fn=print, line_length=100)

    cbs, ckpt, _ = make_callbacks(
        stem, append=bool(initial_epoch), monitor=args.es_monitor,
        patience=args.es_patience,
        start_from_epoch=(args.start_from_epoch if initial_epoch == 0 else 0),
        initial_best=(prior_best if initial_epoch else None),
        schedule=args.lr_schedule)
    cw = (data_pipeline.get_class_weights(args.data_dir, class_names)
          if args.class_weights else None)
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=args.epochs_custom, initial_epoch=initial_epoch,
                     callbacks=cbs, class_weight=cw, verbose=2)
    dt = time.time() - t0
    model.save(config.MODELS_DIR / f"{stem}_final.keras")
    if not ckpt.exists():
        model.save(ckpt)
    loaded = tf.keras.models.load_model(ckpt, compile=False)
    sample = next(iter(val_ds.take(1)))[0][:4]
    parity = check_save_reload(loaded, sample)
    total, trainable = model_zoo.count_params(loaded)

    val_acc = hist.history.get("val_accuracy", [])
    best_new = float(max(val_acc)) if val_acc else float("nan")
    best_val = max(prior_best, best_new) if prior_best else best_new
    epochs_run = initial_epoch + len(hist.history["loss"])
    meta = {
        "run": stem, "task": args.task, "note": args.note,
        "model": "custom_cnn", "backbone": None,
        "data_dir": str(Path(args.data_dir).resolve().name),
        "epochs_planned": args.epochs_custom, "epochs_run": epochs_run,
        "initial_epoch": initial_epoch,
        "best_epoch": _best_epoch(hist.history, initial_epoch),
        "best_val_accuracy": best_val,
        "new_train_seconds": round(dt, 1),
        "prior_train_seconds": round(prior_seconds, 1),
        "train_seconds": round(dt + prior_seconds, 1),
        "total_params": total, "trainable_params": trainable,
        "save_reload_parity": parity, "seed": args.seed,
        "img_size": args.img_size, "batch_size": args.batch_size,
        "optimizer": args.optimizer, "custom_lr": args.custom_lr,
        "head_lr": None, "ft_lr": None,
        "unfreeze_last": None, "lr_schedule": args.lr_schedule,
        "es_monitor": args.es_monitor, "es_patience": args.es_patience,
        "start_from_epoch": args.start_from_epoch,
        "augmented": not args.no_augment,
        "head_only": False,
        "resumed_from": (str(args.resume) if args.resume else None),
        "class_names": class_names,
    }
    (config.LOGS_DIR / f"{stem}_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def train_transfer(args, train_ds, val_ds, class_names, spb: int) -> dict:
    """Two-phase transfer training (head warm-up, then fine-tune)."""
    stem = f"transfer_{args.backbone}_{args.tag}"
    existing = None if args.force else _run_exists(stem)
    if existing:
        return existing
    print(f"=== Model B ({args.backbone}) [{stem}] ===", flush=True)
    model, backbone = model_zoo.build_transfer_model(
        backbone_name=args.backbone,
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=len(class_names),
        learning_rate=args.head_lr,
        optimizer_name=args.optimizer,
        steps_per_epoch=spb, epochs=args.head_epochs,
        schedule=args.lr_schedule)
    model.summary(print_fn=print, line_length=100)
    cw = (data_pipeline.get_class_weights(args.data_dir, class_names)
          if args.class_weights else None)

    cbs1, ckpt, _ = make_callbacks(
        stem, append=False, monitor=args.es_monitor,
        patience=args.es_patience, start_from_epoch=0,
        schedule=args.lr_schedule)
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=args.head_epochs, callbacks=cbs1,
                   class_weight=cw, verbose=2)
    n1 = len(h1.history["loss"])
    best = float(max(h1.history.get("val_accuracy", [float("nan")])))
    best_epoch = _best_epoch(h1.history, 0)

    n2 = 0
    do_ft = (not args.head_only and args.ft_epochs > 0
             and args.unfreeze_last != 0)
    if do_ft:
        model = model_zoo.unfreeze_for_fine_tuning(
            model, backbone, last_n=args.unfreeze_last,
            learning_rate=args.ft_lr, optimizer_name=args.optimizer,
            steps_per_epoch=spb, epochs=args.ft_epochs,
            schedule=args.lr_schedule)
        total, trainable = model_zoo.count_params(model)
        print(f"Fine-tuning: {trainable}/{total} trainable params", flush=True)
        cbs2, ckpt2, _ = make_callbacks(
            stem, append=True, monitor=args.es_monitor,
            patience=args.es_patience, start_from_epoch=0,
            initial_best=best, schedule=args.lr_schedule)
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=n1 + args.ft_epochs, callbacks=cbs2,
                       class_weight=cw, verbose=2, initial_epoch=n1)
        n2 = len(h2.history["loss"])
        if h2.history.get("val_accuracy"):
            phase_best = float(max(h2.history["val_accuracy"]))
            if phase_best > best:
                best = phase_best
                best_epoch = _best_epoch(h2.history, n1)
        ckpt = ckpt2 if ckpt2.exists() else ckpt
    dt = time.time() - t0
    model.save(config.MODELS_DIR / f"{stem}_final.keras")
    if not ckpt.exists():
        model.save(ckpt)
    loaded = tf.keras.models.load_model(ckpt, compile=False)
    sample = next(iter(val_ds.take(1)))[0][:4]
    parity = check_save_reload(loaded, sample)
    total, trainable = model_zoo.count_params(loaded)

    meta = {
        "run": stem, "task": args.task, "note": args.note,
        "model": f"transfer_{args.backbone}", "backbone": args.backbone,
        "data_dir": str(Path(args.data_dir).resolve().name),
        "head_epochs_planned": args.head_epochs,
        "ft_epochs_planned": args.ft_epochs,
        "head_epochs_run": n1, "ft_epochs_run": n2,
        "epochs_run": n1 + n2, "best_epoch": best_epoch,
        "best_val_accuracy": best, "train_seconds": round(dt, 1),
        "total_params": total, "trainable_params": trainable,
        "save_reload_parity": parity, "seed": args.seed,
        "img_size": args.img_size, "batch_size": args.batch_size,
        "optimizer": args.optimizer, "custom_lr": None,
        "head_lr": args.head_lr, "ft_lr": args.ft_lr,
        "unfreeze_last": args.unfreeze_last,
        "lr_schedule": args.lr_schedule,
        "es_monitor": args.es_monitor, "es_patience": args.es_patience,
        "start_from_epoch": 0,
        "head_only": bool(args.head_only or args.unfreeze_last == 0
                          or args.ft_epochs == 0),
        "augmented": not args.no_augment,
        "resumed_from": None,
        "class_names": class_names,
    }
    (config.LOGS_DIR / f"{stem}_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def main() -> None:
    """Entry point: build data, train requested model(s), record metadata."""
    args = parse_args()
    if not args.tag or "/" in args.tag:
        raise ValueError("--tag must be a non-empty filename-safe string")
    data_pipeline.set_seeds(args.seed)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _, class_names = data_pipeline.load_datasets(
        dataset_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size, seed=args.seed,
        use_augmentation=not args.no_augment, subset_train=args.subset)
    spb = steps_per_epoch(train_ds)
    print(f"Classes (label order): {class_names}", flush=True)
    print(f"Data: {args.data_dir} | steps/epoch: {spb} | "
          f"optimizer: {args.optimizer} | schedule: {args.lr_schedule}",
          flush=True)
    metas = []
    if args.model in ("custom", "both"):
        metas.append(train_custom(args, train_ds, val_ds, class_names, spb))
    if args.model in ("transfer", "both"):
        metas.append(train_transfer(args, train_ds, val_ds, class_names, spb))
    print("DONE:", [m["run"] for m in metas], flush=True)


if __name__ == "__main__":
    main()
