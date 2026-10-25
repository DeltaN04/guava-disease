"""Collect every run's metadata into ``outputs/results/experiments.csv``.

Reads ``outputs/logs/*_meta.json`` (written by train.py) and joins the test
metrics from ``outputs/results/*_metrics.json`` when a run was selected for
final evaluation. This is the single machine-readable source for every table
in the report and slides; no number is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

COLUMNS = [
    "run", "task", "data_split", "model", "backbone", "optimizer",
    "head_lr", "ft_lr", "custom_lr", "lr_schedule", "unfreeze_last",
    "epochs", "seed", "best_val_acc", "best_epoch", "test_acc",
    "test_acc_released", "minutes", "params", "trainable", "status", "note",
]


def split_name(data_dir: str | None) -> str:
    """Normalise the data-directory name for tables."""
    if not data_dir:
        return ""
    return "leak-free" if "leakfree" in data_dir else "released"


def main() -> None:
    """Build experiments.csv from meta + metrics JSON files."""
    rows = []
    for p in sorted(config.LOGS_DIR.glob("*_meta.json")):
        if "_smoke" in p.name or "qa" in p.name:
            continue
        meta = json.loads(p.read_text())
        run = meta.get("run", p.name.replace("_meta.json", ""))
        # The old base runs were clobbered by a default-tag smoke test; their
        # metrics JSONs survive but the meta/checkpoint logs are invalid.
        # They are provenance-only and must not enter the experiments table.
        if run in ("custom_base", "transfer_MobileNetV2_base"):
            continue
        metrics_path = config.RESULTS_DIR / f"{run}_metrics.json"
        test_acc = ""
        if metrics_path.exists():
            test_acc = round(json.loads(metrics_path.read_text())["test_accuracy"], 4)
        released_path = config.RESULTS_DIR / f"{run}__released_metrics.json"
        test_acc_released = ""
        if released_path.exists():
            test_acc_released = round(
                json.loads(released_path.read_text())["test_accuracy"], 4)
        status = "completed"
        if "not run" in str(meta.get("note", "")).lower():
            status = "not run"
        rows.append({
            "run": run,
            "task": meta.get("task", "misc"),
            "data_split": split_name(meta.get("data_dir")),
            "model": meta.get("model", ""),
            "backbone": meta.get("backbone", ""),
            "optimizer": meta.get("optimizer", ""),
            "head_lr": meta.get("head_lr", ""),
            "ft_lr": meta.get("ft_lr", ""),
            "custom_lr": meta.get("custom_lr", ""),
            "lr_schedule": meta.get("lr_schedule", ""),
            "unfreeze_last": meta.get("unfreeze_last", ""),
            "epochs": meta.get("epochs_run", ""),
            "seed": meta.get("seed", ""),
            "best_val_acc": (round(meta["best_val_accuracy"], 4)
                             if meta.get("best_val_accuracy") is not None else ""),
            "best_epoch": meta.get("best_epoch", ""),
            "test_acc": test_acc,
            "test_acc_released": test_acc_released,
            "minutes": (round(meta.get("train_seconds", 0) / 60, 1)
                        if meta.get("train_seconds") is not None else ""),
            "params": meta.get("total_params", ""),
            "trainable": meta.get("trainable_params", ""),
            "status": status,
            "note": meta.get("note", ""),
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["task", "run"]).reset_index(drop=True)
    out = config.RESULTS_DIR / "experiments.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} runs)")
    print(df[["run", "task", "data_split", "best_val_acc", "test_acc",
              "minutes"]].to_string(index=False))


if __name__ == "__main__":
    main()
