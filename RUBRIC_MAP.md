# RUBRIC_MAP.md — criterion → exact evidence

Every row points to a file that exists in this repository and to the section,
table, figure, script or log that supplies the evidence. No criterion is
marked complete without a concrete artefact.

| Criterion | What is required | Exact evidence |
|---|---|---|
| **CO1 — Problem definition** | Clear problem statement, motivation, research questions, and positioning against prior work | Report §1 (Introduction) and §2 (Literature review); slides `slides_rubric.pdf` §1 (Problem Explanation and Motivation, frames 3–6); research gap in `report/survey.md` §10; verified 12-study table in `report/survey.md` §8; leakage motivation in `outputs/audit/leakage.json` and `DECISIONS.md` D7. |
| **CO2 — Implementation (models + pipeline)** | Working data pipeline, Model A and Model B implementations, training/evaluation tooling, reproducible runs | `data_pipeline.py` (augmentation, splits, class weights); `models.py` (custom CNN, MobileNetV2/EfficientNetB0/ResNet50, optimiser factory, preprocessing); `train.py` (two-phase training, resumability, metadata); `evaluate.py` (metrics, confusion matrices, ROC, per-image predictions); architecture block diagram in report Figure 1 and slides frame 9–10; smoke-test evidence in `outputs/logs/qa1_meta.json` (save/reload parity true). |
| **CO3 — Implementation (experiments + reproducibility)** | Executed experiments with fixed seeds, logs, resumability, and a machine-readable record | `outputs/results/experiments.csv` (all E1–E6 + T1 runs); `outputs/logs/*_meta.json` (full config, best epoch, train time, params); `outputs/logs/*.csv` (per-epoch history); `scripts/run_b_grid.sh` (E1–E4 launcher); `DECISIONS.md` D11–D16 (protocol, cuts, provenance incident); `PROGRESS.md` phase tracker. |
| **CO4 — Optimisation** | Controlled optimisation study over optimiser, LR, depth, backbone; Model A optimiser/schedule; final configs with seeds; cuts documented | Report §3.4 (grid) and §4.2–4.3 (E1–E6); tables `e1_table.tex`–`e5_table.tex`, `seed_summary.tex`; figures `optimizer_comparison.png`, `lr_sensitivity.png`, `backbone_cost.png`; `outputs/results/experiments.csv`; cuts in `DECISIONS.md` D13 (E5 AdamW/RMSprop not run) and D14 (provenance incident). |
| **CO5 — Results and insights** | Quantitative results with uncertainty, leakage finding, qualitative error analysis, limitations, and honest conclusions | Report §4.1–4.9; `outputs/results/final_results.tex`, `t1_table.tex`, `stats_table.tex`; `outputs/results/stats.json` / `stats.md` (bootstrap CIs, Wilson intervals, exact McNemar); `outputs/plots/accuracy_ci.png`, `roc_*.png`, `*_cm.png`, `gradcam_*contact_sheet.png`; `outputs/results/analysis.md`; limitations in report §4.9 and slides frame 23; leakage control numbers in `outputs/audit/leakage.json` and `outputs/results/final_results.tex`. |

## How to reproduce

```bash
# 1. E1-E4 grid (leak-free validation only; resumable)
bash scripts/run_b_grid.sh

# 2. Task 1 fixed Model A and E5/E6 runs (see PROGRESS.md for exact commands)
./venv/bin/python train.py --model custom --data-dir Dataset --tag es-fixed \
  --epochs-custom 15 --es-monitor val_accuracy --start-from-epoch 12 --task T1

# 3. Collect metadata and generate tables
./venv/bin/python scripts/collect_experiments.py
./venv/bin/python scripts/make_tables.py

# 4. Evaluate final runs (leak-free primary, released secondary)
./venv/bin/python evaluate.py --data-dir Dataset_leakfree \
  --runs custom_final-s42,custom_final-s43,custom_final-s44,transfer_MobileNetV2_final-s42,transfer_MobileNetV2_final-s43,transfer_MobileNetV2_final-s44 \
  --preds
./venv/bin/python evaluate.py --data-dir Dataset \
  --runs custom_final-s42,custom_final-s43,custom_final-s44,transfer_MobileNetV2_final-s42,transfer_MobileNetV2_final-s43,transfer_MobileNetV2_final-s44 \
  --preds --suffix released

# 5. Statistics and figures
./venv/bin/python scripts/stats.py --pairs "custom_final-s42:transfer_MobileNetV2_final-s42"
./venv/bin/python scripts/gradcam.py --ckpt transfer_MobileNetV2_final-s42.keras \
  --data-dir Dataset_leakfree --label "B final (MobileNetV2)"
./venv/bin/python scripts/make_figures.py

# 6. Compile report and slides
cd report && ../bin/tectonic report.tex && ../bin/tectonic slides_rubric.tex
```

## Status

All five criteria have evidence in this repository. The only explicit cuts are
E5 AdamW/RMSprop (CPU budget; `DECISIONS.md` D13), the 15-epoch cap on the
released fixed-Model-A run (task requires at least 15; `DECISIONS.md` D11), and
the provenance-only status of the original base runs after the smoke-test
clobber (`DECISIONS.md` D14). These are documented as cuts, not hidden.
