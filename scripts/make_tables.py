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
        "\\midrule\n" + body + " \\\\\n\\bottomrule\n\\end{tabular}\n"))


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
        "\\midrule\n" + body + " \\\\\n\\bottomrule\n\\end{tabular}\n"))


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
        acc_sd = f"{acc.std(ddof=1):.4f}" if len(g) > 1 else "--"
        f1_sd = f"{f1.std(ddof=1):.4f}" if len(g) > 1 else "--"
        rows.append((tex(group), split, f"{acc.mean():.4f} $\\pm$ "
                     f"{acc_sd}", f"{f1.mean():.4f} $\\pm$ "
                     f"{f1_sd}", f"{len(g)}"))
    body = "\\\\\n".join(" & ".join(x) for x in rows)
    write("seed_summary.tex", (
        "\\begin{tabular}{llccc}\n\\toprule\n"
        "Final config & Split & Test acc mean $\\pm$ std & "
        "Best val acc mean $\\pm$ std & Seeds \\\\\n\\midrule\n"
        + body + " \\\\\n\\bottomrule\n\\end{tabular}\n"))


def grid_table(exp: pd.DataFrame, task: str, name: str,
               title: str) -> None:
    """One optimisation-grid table (E1-E5)."""
    df = exp[exp["task"] == task].copy()
    if task in ("E3", "E4"):
        # E3/E4 are 6+6 runs; the matched baseline is the E2 reference, not
        # the 8+8 E1 Adam run (which would confound budget with depth/backbone).
        ref = exp[exp["run"] == "transfer_MobileNetV2_e2-ref"]
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
        "\\midrule\n" + body + " \\\\\n\\bottomrule\n\\end{tabular}\n"))


def _lr(x: object) -> str:
    """Compact scientific notation for learning rates."""
    mant, exp = f"{float(x):.0e}".split("e")
    return f"{mant}e{int(exp)}"


def grid_all_table(exp: pd.DataFrame) -> None:
    """One compact validation grid for E1--E5, with matched references."""
    sections = [
        ("E1: Model B optimiser (8+8 epochs)", "E1",
         ["transfer_MobileNetV2_e1-adam",
          "transfer_MobileNetV2_e1-adamw",
          "transfer_MobileNetV2_e1-rmsprop",
          "transfer_MobileNetV2_e1-sgd"],
         lambda r: {"adam": "Adam", "adamw": "AdamW",
                    "rmsprop": "RMSprop", "sgd": "SGD"}[str(r["optimizer"])]),
        ("E2: learning rate, one factor at a time (6+6)", "E2",
         ["transfer_MobileNetV2_e2-ref",
          "transfer_MobileNetV2_e2-head1e2",
          "transfer_MobileNetV2_e2-head3e4",
          "transfer_MobileNetV2_e2-ft1e4",
          "transfer_MobileNetV2_e2-ft1e6"],
         lambda r: (f"head {_lr(r['head_lr'])}, ft {_lr(r['ft_lr'])}"
                    + (" (ref)" if str(r["run"]).endswith("e2-ref") else ""))),
        ("E3: fine-tune depth (6+6; reference = E2 ref, 0.9789)", "E3",
         ["transfer_MobileNetV2_e3-unfreeze0",
          "transfer_MobileNetV2_e3-unfreeze60",
          "transfer_MobileNetV2_e3-unfreezeall"],
         lambda r: {"0": "head only", "60": "last 60",
                    "9999": "all non-BN"}[str(int(float(r["unfreeze_last"])))]),
        ("E4: backbone (6+6)", "E4",
         ["transfer_EfficientNetB0_e4-effnet",
          "transfer_MobileNetV2_e2-ref",
          "transfer_ResNet50_e4-resnet"],
         lambda r: ("EfficientNetB0" if r["backbone"] == "EfficientNetB0"
                    else "MobileNetV2 (E2 ref)" if str(r["run"]).endswith("e2-ref")
                    else "ResNet50")),
        ("E5: Model A optimiser/schedule (8 epochs)", "E5",
         ["custom_e5-adam", "custom_e5-adam-cosine", "custom_e5-sgd"],
         lambda r: {"adam": "Adam", "sgd": "SGD"}.get(
             str(r["optimizer"]), str(r["optimizer"]))
                   + (" + cosine" if str(r["lr_schedule"]) == "cosine" else "")),
    ]
    rows = []
    for title, _task, order, label in sections:
        rows.append(f"\\addlinespace\n\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}} ")
        for run in order:
            r = exp[exp["run"] == run]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(" & ".join((label(r), fmt(r["best_val_acc"]),
                                    f"{int(r['best_epoch'])}",
                                    f"{float(r['minutes']):.1f}")))
    body = "\\\\\n".join(rows)
    write("grid_all.tex", (
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Setting & Best val acc & Best epoch & Min \\\\\n\\midrule\n"
        + body + " \\\\\n\\bottomrule\n\\end{tabular}\n"))


def short_run(name: str) -> str:
    """Short display name for wide tables."""
    return (name.replace("transfer_MobileNetV2_", "B: ")
                .replace("transfer_EfficientNetB0_", "B: ")
                .replace("transfer_ResNet50_", "B: ")
                .replace("custom_", "A: "))


def stats_table() -> None:
    """McNemar table from stats.json."""
    p = config.RESULTS_DIR / "stats.json"
    if not p.exists():
        write("stats_table.tex", "% no stats yet\n")
        return
    stats = json.loads(p.read_text())
    rows = []
    for pair in stats.get("pairs", []):
        verdict = pair["verdict"].replace("within noise at alpha=0.05",
                                          "within noise")
        rows.append((
            tex(short_run(pair["a"])), tex(short_run(pair["b"])),
            f"{pair['n_pairs']}",
            f"{pair['accuracy_a']:.4f}", f"{pair['accuracy_b']:.4f}",
            f"{pair['delta_b_minus_a']:+.4f}",
            f"{pair['a_wrong_b_correct']}/{pair['a_correct_b_wrong']}",
            f"{pair['p_exact']:.4f}", tex(verdict)))
    body = "\\\\\n".join(" & ".join(x) for x in rows)
    write("stats_table.tex", (
        "\\begin{tabular}{llcccccc l}\n\\toprule\n"
        "A & B & n & Acc A & Acc B & $\\Delta$ & b/c & "
        "McNemar $p$ & Verdict \\\\\n"
        "\\midrule\n" + body + " \\\\\n\\bottomrule\n\\end{tabular}\n"))


def ci_table() -> None:
    """Per-run bootstrap and Wilson intervals for final runs."""
    p = config.RESULTS_DIR / "stats.json"
    if not p.exists():
        write("ci_table.tex", "% no stats yet\n")
        return
    stats = json.loads(p.read_text())
    rows = []
    for run, r in stats.get("runs", {}).items():
        a_lo, a_hi = r["accuracy_boot_ci95"]
        f_lo, f_hi = r["macro_f1_boot_ci95"]
        w_lo, w_hi = r["accuracy_wilson_ci95"]
        rows.append((tex(short_run(run)), f"{r['n']}",
                     f"{r['accuracy']:.4f}",
                     f"[{a_lo:.4f}, {a_hi:.4f}]",
                     f"{r['macro_f1']:.4f}",
                     f"[{f_lo:.4f}, {f_hi:.4f}]",
                     f"[{w_lo:.4f}, {w_hi:.4f}]"))
    body = "\\\\\n".join(" & ".join(x) for x in rows)
    write("ci_table.tex", (
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Run & n & Acc & 95\\% boot CI & Macro F1 & 95\\% boot CI & "
        "Wilson CI \\\\\n\\midrule\n"
        + body + " \\\\\n\\bottomrule\n\\end{tabular}\n"))


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
    grid_all_table(exp)
    stats_table()
    ci_table()


if __name__ == "__main__":
    main()
