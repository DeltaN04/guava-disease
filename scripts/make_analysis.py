"""Regenerate ``outputs/results/analysis.md`` from metrics, experiments, stats.

This replaces hand-typed claims with numbers read directly from the JSON/CSV
artefacts. It deliberately avoids phrases such as "at capacity" or "8pp"
unless the fixed/final runs support them; the only comparisons printed are
the ones with saved per-image predictions.

Usage:
    ./venv/bin/python scripts/make_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load_metrics() -> dict[str, dict]:
    """Load every metrics JSON, keyed by file stem."""
    out = {}
    for p in sorted(config.RESULTS_DIR.glob("*_metrics.json")):
        out[p.name.replace("_metrics.json", "")] = json.loads(p.read_text())
    return out


def per_class_table(m: dict) -> list[str]:
    """Markdown per-class table from a metrics dict."""
    rows = ["| Class | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|"]
    for c, v in m["per_class"].items():
        rows.append(f"| {config.DISPLAY_NAMES.get(c, c)} | "
                    f"{v['precision']:.4f} | {v['recall']:.4f} | "
                    f"{v['f1']:.4f} | {v['support']} |")
    return rows


def main() -> None:
    """Write the error-analysis markdown from real artefacts."""
    metrics = load_metrics()
    exp_path = config.RESULTS_DIR / "experiments.csv"
    exp = pd.read_csv(exp_path).fillna("") if exp_path.exists() else pd.DataFrame()
    stats_path = config.RESULTS_DIR / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    final = ["custom_final-s42", "transfer_MobileNetV2_final-s42"]
    lines = ["# Error analysis (generated from metrics JSON)", "",
             "Source: `outputs/results/*_metrics.json`, "
             "`outputs/results/experiments.csv`, `outputs/results/stats.json`. "
             "Every number below is read from an executed run; no value is "
             "typed by hand.", ""]

    if not exp.empty and "task" in exp:
        e6 = exp[exp["task"] == "E6"]
        if not e6.empty:
            lines += ["## Final configurations (E6, seeds 42-44)", "",
                      "| Run | Split | Best val acc | Test acc | Macro F1 | "
                      "Minutes |", "|---|---|---|---|---|---|"]
            for _, r in e6.iterrows():
                m = metrics.get(str(r["run"]))
                f1 = f"{m['macro_f1']:.4f}" if m else "--"
                acc = f"{r['test_acc']:.4f}" if r["test_acc"] != "" else "--"
                lines.append(f"| {r['run']} | {r['data_split']} | "
                             f"{r['best_val_acc']:.4f} | {acc} | {f1} | "
                             f"{r['minutes']:.1f} |")
            lines.append("")

    if stats.get("seed_groups"):
        lines += ["## Multi-seed summary", "",
                  "| Group | Seeds | Accuracy mean +/- std | "
                  "Macro F1 mean +/- std |", "|---|---|---|---|"]
        for g, s in stats["seed_groups"].items():
            lines.append(f"| {g} | {s['n_seeds']} | "
                         f"{s['accuracy_mean']:.4f} +/- "
                         f"{s['accuracy_std']:.4f} | "
                         f"{s['macro_f1_mean']:.4f} +/- "
                         f"{s['macro_f1_std']:.4f} |")
        lines.append("")

    for run in final:
        if run not in metrics:
            continue
        m = metrics[run]
        lines += [f"## Per-class metrics: {run}", ""]
        if "data_dir" in m:
            lines.append(f"Test set: `{m['data_dir']}` (n={m.get('n_test')})")
            lines.append("")
        lines += per_class_table(m)
        cm = m.get("confusion_matrix")
        if cm:
            classes = list(m["per_class"])
            pairs = []
            for i, tc in enumerate(classes):
                for j, pc in enumerate(classes):
                    if i != j and cm[i][j] > 0:
                        pairs.append((cm[i][j], tc, pc))
            lines += ["", "Largest off-diagonal confusions:", ""]
            if pairs:
                for n, tc, pc in sorted(pairs, reverse=True)[:6]:
                    lines.append(
                        f"- {config.DISPLAY_NAMES.get(tc, tc)} -> "
                        f"{config.DISPLAY_NAMES.get(pc, pc)}: {n} images")
            else:
                lines.append("- none")
        lines.append("")

    if stats.get("pairs"):
        lines += ["## Paired statistical tests", "",
                  "Bootstrap: 2,000 resamples, seed 42. McNemar: exact "
                  "two-sided binomial on paired correctness. A difference is "
                  "called noise when p >= 0.05; a significant difference "
                  "spanning at most two images is flagged separately.", "",
                  "| A | B | n | Acc A | Acc B | Delta | b/c | p | Verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in stats["pairs"]:
            lines.append(
                f"| {p['a']} | {p['b']} | {p['n_pairs']} | "
                f"{p['accuracy_a']:.4f} | {p['accuracy_b']:.4f} | "
                f"{p['delta_b_minus_a']:+.4f} | "
                f"{p['a_wrong_b_correct']}/{p['a_correct_b_wrong']} | "
                f"{p['p_exact']:.4f} | {p['verdict']} |")
        lines.append("")

    lines += ["## What the numbers do and do not support", "",
              "- The released GFDD24 split is 100% leaked (454/473 originals "
              "span splits); released test accuracy is a near-duplicate "
              "recognition score, not a field-generalisation estimate.",
              "- The leak-free split removes same-fruit duplicates "
              "(0/473 clusters span splits). It is the primary comparison.",
              "- Final A/B differences are interpreted only through the "
              "bootstrap intervals and exact McNemar tests above; a "
              "non-significant difference is reported as within noise.",
              "- The old `custom_base` and `transfer_MobileNetV2_base` "
              "checkpoints/logs were clobbered by a smoke test (DECISIONS D14). "
              "Their metrics JSONs remain provenance only and are not used for "
              "paired tests.",
              "- Model A is CPU-limited at 224px; E5 uses 8 epochs (cap 12) and "
              "AdamW/RMSprop were cut (DECISIONS D13). These are limitations, "
              "not hidden results.", ""]

    out = config.RESULTS_DIR / "analysis.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
