"""Generate LaTeX tables from real experiment metadata and prediction stats.

Every number in these tables comes from ``outputs/results/experiments.csv``
(joined meta + test metrics) and ``outputs/results/stats.json``. Run this
after training/evaluation; the report and slides ``\\input`` the outputs.

Usage:
    ./venv/bin/python scripts/make_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def tex(s: object) -> str:
    """Escape underscores for LaTeX text mode."""
    return str(s).replace("_", r"\_")


def fmt(x: object, nd: int = 4) -> str:
    """Format a numeric cell, leaving blanks for missing values."""
    if x is None or (isinstance(x, float) and x != x) or x == "":
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return tex(x)


def write(name: str, text: str) -> None:
    """Write a LaTeX snippet and print its path."""
    p = config.RESULTS_DIR / name
    p.write_text(text)
    print(f"Wrote {p}")


def final_table(exp: pd.DataFrame) -> None:
    """Per-run final results (T1 + E6), including both test splits."""
    df = exp[exp["task"].isin(["T1", "E6"])].copy()
    if df.empty:
        write("final_results.tex", "% no final runs yet\n")
        return
    rows = []
    for _, r in df.iterrows():
        for split, col in (("leak-free", "test_acc"),
                           ("released", "test_acc_released")):
            if r.get(col, "") == "" or pd.isna(r.get(col)):
                continue
            rows.append((tex(r["run"]), split, fmt(r[col]),
                         fmt(r["best_val_acc"]), f"{int(r['epochs'])}",
                         f"{float(r['minutes']):.1f}",
                         f"{int(r['params']):,}"))
    body = "\\\\\n".join(" & ".join(x) for x in rows)
    write("final_results.tex", (
        "\\begin{tabular}{llcccrr}\n\\toprule\n"
        "Run & Test split & Test acc & Val acc & Epochs & Min & Params \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"))


def t1_table(exp: pd.DataFrame) -> None:
    """Task 1 fixed-Model-A table (released + leak-free rows)."""
    df = exp[exp["task"] == "T1"].copy()
    if df.empty:
        write("t1_table.tex", "% no T1 runs yet\n")
        return
    rows = []
    for _, r in df.iterrows():
        for split, col in (("leak-free", "test_acc"),
                           ("released", "test_acc_released")):
            if r.get(col, "") == "" or pd.isna(r.get(col)):
                continue
            rows.append((tex(r["run"]), split, fmt(r[col]),
                         fmt(r["best_val_acc"]), f"{int(r['epochs'])}",
                         f"{float(r['minutes']):.1f}"))
    body = "\\\\\n".join(" & ".join(x) for x in rows)
    write("t1_table.tex", (
        "\\begin{tabular}{llcccc}\n\\toprule\n"
        "Run & Test split & Test acc & Val acc & Epochs & Min \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"))


def seed_summary(exp: pd.DataFrame) -> None:
    """Mean +/- std across seeds 42/43/44 for final configs."""
    df = exp[(exp["task"] == "E6") & (exp["test_acc"] != "")].copy()
    if df.empty:
        write("seed_summary.tex", "% no multi-seed final runs yet\n")
        return
    df["group"] = df["run"].str.rsplit("-s", n=1).str[0]
    rows = []
    for (group, split), g in df.groupby(["group", "data_split"]):
        acc = g["test_acc"].astype(float)
        f1 = g["best_val_acc"].astype(float)
        rows.append((tex(group), split, f"{acc.mean():.4f} $\\pm$ "
                     f"{acc.std(ddof=1):.4f}", f"{f1.mean():.4f} $\\pm$ "
                     f"{f1.std(ddof=1):.4f}", f"{len(g)}"))
    body = "\\\\\n".join(" & ".join(x) for x in rows)
    write("seed_summary.tex", (
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "Final config & Split & Test acc mean $\\pm$ std & "
        "Best val acc mean $\\pm$ std & Seeds \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n"))


def grid_table(exp: pd.DataFrame, task: str, name: str,
               title: str) -> None:
    """One optimisation-grid table (E1-E5)."""
    df = exp[exp["task"] == task].copy()
    if task in ("E2", "E3", "E4"):
        ref = exp[exp["run"] == "transfer_MobileNetV2_e1-adam"]
        if not ref.empty:
            df = pd.concat([ref, df], ignore_index=True)
    if df.empty:
        write(name, f"% no {task} runs yet\n")
        return
    rows = []
    for _, r in df.sort_values("run").iterrows():
        cfg = r["run"]
        cfg = cfg.replace("transfer_MobileNetV2_", "").replace("custom_", "")
        extra = []
        if r.get("optimizer"):
            extra.append(str(r["optimizer"]))
        if task == "E1":
            pass
        elif task == "E2":
            extra.append(f"head={r['head_lr']} ft={r['ft_lr']}")
        elif task == "E3":
            extra.append(f"unfreeze={r['unfreeze_last']}")
        elif task == "E4":
            extra.append(str(r["backbone"]))
        elif task == "E5":
            extra.append(f"{r['optimizer']}/{r['lr_schedule']}")
        rows.append((tex(cfg), tex("; ".join(extra)),
                     fmt(r["best_val_acc"]), f"{int(r['best_epoch'])}",
                     f"{float(r['minutes']):.1f}"))
    body = "\\\\\n".join(" & ".join(x) for x in rows)
    write(name, (
        "\\begin{tabular}{llccc}\n\\toprule\n"
        f"{title} & Config & Best val acc & Best epoch & Min \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"))


def stats_table() -> None:
    """McNemar + bootstrap CI table from stats.json."""
    p = config.RESULTS_DIR / "stats.json"
    if not p.exists():
        write("stats_table.tex", "% no stats yet\n")
        return
    stats = json.loads(p.read_text())
    rows = []
    for pair in stats.get("pairs", []):
        rows.append((
            tex(pair["a"]), tex(pair["b"]), f"{pair['n_pairs']}",
            f"{pair['accuracy_a']:.4f}", f"{pair['accuracy_b']:.4f}",
            f"{pair['delta_b_minus_a']:+.4f}",
            f"{pair['a_wrong_b_correct']}/{pair['a_correct_b_wrong']}",
            f"{pair['p_exact']:.4f}", tex(pair["verdict"])))
    body = "\\\\\n".join(" & ".join(x) for x in rows)
    write("stats_table.tex", (
        "\\begin{tabular}{llcccccc}\n\\toprule\n"
        "A & B & n & Acc A & Acc B & $\\Delta$ & b/c & McNemar $p$ \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"))


def main() -> None:
    """Generate all table snippets."""
    exp_path = config.RESULTS_DIR / "experiments.csv"
    if not exp_path.exists():
        raise FileNotFoundError("run scripts/collect_experiments.py first")
    exp = pd.read_csv(exp_path).fillna("")
    t1_table(exp)
    final_table(exp)
    seed_summary(exp)
    grid_table(exp, "E1", "e1_table.tex", "Optimiser")
    grid_table(exp, "E2", "e2_table.tex", "LR setting")
    grid_table(exp, "E3", "e3_table.tex", "Fine-tune depth")
    grid_table(exp, "E4", "e4_table.tex", "Backbone")
    grid_table(exp, "E5", "e5_table.tex", "Model A setting")
    stats_table()


if __name__ == "__main__":
    main()
