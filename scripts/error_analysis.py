"""Generate per-class metrics and confusion details from real metrics JSON.

Reads ``outputs/results/*_metrics.json`` and writes
``outputs/results/per_class_f1.md``. This is the source of truth for the
error-analysis text; no per-class number should be typed by hand.

Usage:
    ./venv/bin/python scripts/error_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main() -> None:
    """Write per-class P/R/F1 and the largest confusion pairs."""
    runs = []
    for p in sorted(config.RESULTS_DIR.glob("*_metrics.json")):
        runs.append((p.name.replace("_metrics.json", ""),
                     json.loads(p.read_text())))
    if not runs:
        raise FileNotFoundError("no metrics JSON; run evaluate.py first")
    lines = ["# Per-class metrics (generated from *_metrics.json)", ""]
    for run, m in runs:
        lines.append(f"## {run}")
        if "data_dir" in m:
            lines.append(f"Test set: `{m['data_dir']}` ($n={m.get('n_test')}$)")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for c, v in m["per_class"].items():
            lines.append(f"| {config.DISPLAY_NAMES.get(c, c)} | "
                         f"{v['precision']:.4f} | {v['recall']:.4f} | "
                         f"{v['f1']:.4f} | {v['support']} |")
        cm = m.get("confusion_matrix")
        if cm:
            classes = list(m["per_class"])
            lines.append("")
            lines.append("Largest off-diagonal confusions:")
            pairs = []
            for i, true_c in enumerate(classes):
                for j, pred_c in enumerate(classes):
                    if i != j and cm[i][j] > 0:
                        pairs.append((cm[i][j], true_c, pred_c))
            for n, true_c, pred_c in sorted(pairs, reverse=True)[:6]:
                lines.append(f"- {config.DISPLAY_NAMES.get(true_c, true_c)} "
                             f"-> {config.DISPLAY_NAMES.get(pred_c, pred_c)}: "
                             f"{n} images")
        lines.append("")
    out = config.RESULTS_DIR / "per_class_f1.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
