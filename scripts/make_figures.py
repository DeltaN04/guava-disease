"""Phase 4: slide-ready summary figures from evaluation outputs.

Reads ``outputs/results/*_metrics.json``, ``results_table.csv`` and
``outputs/logs/*_meta.json``; writes to ``outputs/plots/`:
  * ``params_compare.png`` — total vs trainable parameter bars per run
  * ``f1_per_class.png`` — grouped per-class F1 bars (main models)
  * ``time_compare.png`` — training minutes per run

Usage:
    ./venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from evaluate import RUN_LABELS  # noqa: E402


def load_metrics() -> dict:
    """Load per-run lite metrics JSON files from results dir."""
    out = {}
    for p in sorted(config.RESULTS_DIR.glob("*_metrics.json")):
        out[p.name.replace("_metrics.json", "")] = json.loads(p.read_text())
    return out


def main() -> None:
    """Build parameter, F1 and training-time comparison figures."""
    metrics = load_metrics()
    print(f"Runs with metrics: {list(metrics)}", flush=True)
    if not metrics:
        print("No metrics found; run evaluate.py first.", flush=True)
        return
    runs = [r for r in RUN_LABELS if r in metrics] or list(metrics)
    short = [RUN_LABELS.get(r, r) for r in runs]

    metas = {}
    for r in runs:
        p = config.LOGS_DIR / f"{r}_meta.json"
        metas[r] = json.loads(p.read_text()) if p.exists() else {}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(runs))
    totals = [metas[r].get("total_params", 0) / 1e6 for r in runs]
    trains = [metas[r].get("trainable_params", 0) / 1e6 for r in runs]
    ax.bar(x - 0.2, totals, 0.4, label="Total params (M)")
    ax.bar(x + 0.2, trains, 0.4, label="Trainable params (M)")
    ax.set_xticks(x, short, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Millions of parameters")
    ax.set_title("Model size: total vs trainable parameters")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "params_compare.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    classes = list(next(iter(metrics.values()))["per_class"])
    x = np.arange(len(classes))
    mains = [r for r in ("custom_base", "transfer_MobileNetV2_base")
             if r in metrics] or runs[:2]
    for k, r in enumerate(mains):
        f1 = [metrics[r]["per_class"][c]["f1"] for c in classes]
        bars = ax.bar(x + (k - (len(mains) - 1) / 2) * 0.35, f1, 0.35,
                      label=RUN_LABELS.get(r, r))
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [config.DISPLAY_NAMES.get(c, c) for c in classes])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 on the test set (main models)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "f1_per_class.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    mins = [metas[r].get("train_seconds", 0) / 60 for r in runs]
    bars = ax.bar(short, mins)
    ax.bar_label(bars, fmt="%.0f min", fontsize=8)
    ax.set_ylabel("Minutes")
    ax.set_title("Wall-clock training time per run (CPU)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "time_compare.png", dpi=150)
    plt.close(fig)
    print(f"Wrote summary figures to {config.PLOTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
