"""Paired statistics for final runs: bootstrap CIs, Wilson intervals, McNemar.

Reads per-image predictions written by ``evaluate.py --preds``:
``outputs/results/<run>_preds.csv`` with columns ``file, y_true, y_pred``.

Outputs ``outputs/results/stats.json``, ``stats.md`` and
``outputs/plots/accuracy_ci.png``.

Usage:
    ./venv/bin/python scripts/stats.py \
        --pairs "custom_final-s42:transfer_MobileNetV2_final-s42,..."
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

BOOTSTRAP_N = 2000
RNG = np.random.default_rng(config.SEED)


def load_preds() -> dict[str, pd.DataFrame]:
    """Load every per-image prediction CSV, indexed by run stem."""
    out = {}
    for p in sorted(config.RESULTS_DIR.glob("*_preds.csv")):
        run = p.name.replace("_preds.csv", "")
        out[run] = pd.read_csv(p)
    return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int = BOOTSTRAP_N) -> dict:
    """Percentile bootstrap 95% CI for accuracy and macro-F1."""
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro",
                            zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_boot_ci95": [float(np.percentile(accs, 2.5)),
                               float(np.percentile(accs, 97.5))],
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "macro_f1_boot_ci95": [float(np.percentile(f1s, 2.5)),
                               float(np.percentile(f1s, 97.5))],
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for a binomial proportion."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def mcnemar(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Exact McNemar test on paired correctness over identical test images."""
    merged = a[["file", "y_true", "correct"]].merge(
        b[["file", "y_true", "correct"]], on=["file", "y_true"],
        suffixes=("_a", "_b"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("prediction files do not cover identical images")
    b01 = int(((merged.correct_a == 0) & (merged.correct_b == 1)).sum())
    b10 = int(((merged.correct_a == 1) & (merged.correct_b == 0)).sum())
    n_disc = b01 + b10
    if n_disc == 0:
        p_exact = 1.0
        stat = 0.0
        p_chi = 1.0
    else:
        p_exact = float(binomtest(b01, n_disc, 0.5,
                                  alternative="two-sided").pvalue)
        stat = (abs(b01 - b10) - 1) ** 2 / n_disc
        p_chi = float(chi2.sf(stat, df=1))
    return {
        "n_pairs": int(len(merged)),
        "a_correct_b_wrong": b10, "a_wrong_b_correct": b01,
        "discordant": n_disc, "mcnemar_chi2": float(stat),
        "p_exact": p_exact, "p_chi2_continuity": p_chi,
    }


def main() -> None:
    """Compute CIs and McNemar tests for the final runs."""
    ap = argparse.ArgumentParser(description="Paired statistics.")
    ap.add_argument("--pairs", default="",
                    help="comma-separated runA:runB pairs")
    ap.add_argument("--runs", default="",
                    help="comma-separated runs for CI figure (default: all)")
    args = ap.parse_args()
    preds = load_preds()
    if not preds:
        raise FileNotFoundError("No *_preds.csv; run evaluate.py --preds first")
    wanted = ([s.strip() for s in args.runs.split(",") if s.strip()]
              if args.runs else list(preds))
    missing = [r for r in wanted if r not in preds]
    if missing:
        raise FileNotFoundError(f"missing preds: {missing}")

    results: dict = {"runs": {}, "pairs": [], "seed_groups": {}}
    for run in wanted:
        df = preds[run]
        y_true = df["y_true"].to_numpy()
        y_pred = df["y_pred"].to_numpy()
        k = int((y_true == y_pred).sum())
        n = int(len(df))
        rec = bootstrap_ci(y_true, y_pred)
        rec.update({"n": n, "correct": k,
                    "accuracy_wilson_ci95": wilson_interval(k, n),
                    "test_set": str(df["test_set"].iloc[0])})
        results["runs"][run] = rec

    pairs = [p for p in args.pairs.split(",") if p.strip()]
    if not pairs:
        runs = list(results["runs"])
        for a in runs:
            for b in runs:
                if (a < b and results["runs"][a]["test_set"]
                        == results["runs"][b]["test_set"]):
                    pairs.append(f"{a}:{b}")
    for pair in pairs:
        a, b = [s.strip() for s in pair.split(":")]
        if a not in results["runs"] or b not in results["runs"]:
            print(f"SKIP pair {pair}: missing preds")
            continue
        test = mcnemar(preds[a], preds[b])
        diff = results["runs"][b]["accuracy"] - results["runs"][a]["accuracy"]
        imgs = abs(diff) * results["runs"][a]["n"]
        sig = test["p_exact"] < 0.05
        if not sig:
            verdict = "within noise at alpha=0.05"
        elif imgs <= 2:
            verdict = "significant but within 1-2 images"
        else:
            verdict = "significant"
        results["pairs"].append({
            "a": a, "b": b, "test_set": results["runs"][a]["test_set"],
            "accuracy_a": results["runs"][a]["accuracy"],
            "accuracy_b": results["runs"][b]["accuracy"],
            "delta_b_minus_a": float(diff),
            "delta_images": float(imgs),
            "verdict": verdict, **test,
        })

    # Multi-seed groups: run names must share a prefix before "-s<seed>"
    groups: dict[str, list[str]] = defaultdict(list)
    for run in results["runs"]:
        if "-s" in run:
            groups[run.rsplit("-s", 1)[0]].append(run)
    for group, runs in groups.items():
        accs = [results["runs"][r]["accuracy"] for r in runs]
        f1s = [results["runs"][r]["macro_f1"] for r in runs]
        if len(runs) > 1:
            results["seed_groups"][group] = {
                "runs": runs, "n_seeds": len(runs),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs, ddof=1)),
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s, ddof=1)),
            }

    (config.RESULTS_DIR / "stats.json").write_text(
        json.dumps(results, indent=2))

    lines = ["# Paired statistics (final runs, real predictions)", "",
             "Bootstrap: 2000 resamples, seed 42, percentile 95% CI. "
             "McNemar: exact binomial two-sided on paired correctness.", "",
             "| Run | Test set | n | Acc | 95% bootstrap CI | Wilson CI | Macro F1 |",
             "|---|---|---|---|---|---|---|"]
    for run, r in results["runs"].items():
        lines.append(
            f"| {run} | {r['test_set']} | {r['n']} | {r['accuracy']:.4f} | "
            f"[{r['accuracy_boot_ci95'][0]:.4f}, "
            f"{r['accuracy_boot_ci95'][1]:.4f}] | "
            f"[{r['accuracy_wilson_ci95'][0]:.4f}, "
            f"{r['accuracy_wilson_ci95'][1]:.4f}] | "
            f"{r['macro_f1']:.4f} |")
    lines += ["", "## Paired comparisons", "",
              "| A | B | n | Acc A | Acc B | Delta (B-A) | b/c | McNemar p | Verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in results["pairs"]:
        lines.append(
            f"| {p['a']} | {p['b']} | {p['n_pairs']} | "
            f"{p['accuracy_a']:.4f} | {p['accuracy_b']:.4f} | "
            f"{p['delta_b_minus_a']:+.4f} | "
            f"{p['a_wrong_b_correct']}/{p['a_correct_b_wrong']} | "
            f"{p['p_exact']:.4f} | {p['verdict']} |")
    if results["seed_groups"]:
        lines += ["", "## Multi-seed final configurations", "",
                  "| Group | Seeds | Acc mean +/- std | Macro F1 mean +/- std |",
                  "|---|---|---|---|"]
        for g, s in results["seed_groups"].items():
            lines.append(
                f"| {g} | {s['n_seeds']} | {s['accuracy_mean']:.4f} +/- "
                f"{s['accuracy_std']:.4f} | {s['macro_f1_mean']:.4f} +/- "
                f"{s['macro_f1_std']:.4f} |")
    (config.RESULTS_DIR / "stats.md").write_text("\n".join(lines) + "\n")

    runs = list(results["runs"])
    accs = [results["runs"][r]["accuracy"] for r in runs]
    lo = [results["runs"][r]["accuracy"] - results["runs"][r]["accuracy_boot_ci95"][0]
          for r in runs]
    hi = [results["runs"][r]["accuracy_boot_ci95"][1] - results["runs"][r]["accuracy"]
          for r in runs]
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(runs)), 4.5))
    x = np.arange(len(runs))
    ax.bar(x, accs, yerr=[lo, hi], capsize=4, color="#2E75B6")
    for i, v in enumerate(accs):
        ax.text(i, min(1.01, v + 0.012), f"{v:.3f}", ha="center", fontsize=8)
    def _short(run: str) -> str:
        return (run.replace("custom_final-s", "A s")
                   .replace("transfer_EfficientNetB0_final-s", "B s")
                   .replace("__released", " (rel)"))
    ax.set_xticks(x, [_short(r) for r in runs], rotation=0, fontsize=8)
    ax.set_ylim(max(0.0, min(accs) - 0.05), 1.02)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Final runs: accuracy with 95% bootstrap CI")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "accuracy_ci.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {config.RESULTS_DIR / 'stats.json'}, stats.md and "
          f"{config.PLOTS_DIR / 'accuracy_ci.png'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
