# PROGRESS.md — phase tracker (updated after every phase)

| Phase | Status | Key numbers / notes |
|---|---|---|
| 0 Recon + setup | DONE | macOS 26.5 arm64, Py3.12, TF 2.21 CPU-only, Tectonic 0.17.0; venv + requirements installed; dataset staged at `Dataset/` |
| 1 Dataset audit | DONE | 3784 files, 0 corrupt, all 512x512 RGB PNG; val=755 on disk vs 775 expected (discrepancy reported); **leakage: 100% of val+test share an original with train (454/473 originals span splits)**; `Dataset_leakfree/` built (train 2632 / val 760 / test 392, 0/473 clusters span) |
| 2 Implementation + smoke test | DONE (bugfix 9 Oct) | custom CNN + MobileNetV2 transfer (two-phase) implemented; **train/val skew bug found+fixed** (`value_range=(0,1)`, D8); post-fix smoke save/reload parity True |
| 3 Original experiments | DONE (provenance) | A base 9ep val 0.9364 (31.6min), B base 15+20 val 0.9960 (14.5min); frozen 15ep 0.9907, noaug 15+18 0.9974; leakfree: A-partial 16/40 best 0.9737, B 15+17 0.9855. **Base meta/checkpoints clobbered by a later smoke test (D14); metrics JSONs survive and are provenance only.** |
| 4 Evaluation + analysis | DONE (old runs) | test: A 0.9110/F1 0.9173/AUC 0.9949, B 0.9921/0.9911/0.9999, frozen 0.9948, noaug 0.9948; CMs, ROC, curves, Grad-CAM; `outputs/results/analysis.md` |
| 5 Research survey | DONE | 12-study table + families + gap in `survey.md`; 22-entry `references.bib`; all verified live; Pratama 2025 re-verified via Crossref 25 Oct 2026 (D15) |
| 6 First Beamer deck | DONE (superseded) | 18 frames, Madrid navy/blue, 0 warnings, page-by-page PNG inspection; kept as `slides.tex/pdf` |
| 7 Final QA (old deck) | DONE | smoke re-passed; no placeholders; README/analysis/slides committed at `d72a501` |
| 8 Final experiments (branch `final-experiments`) | IN PROGRESS | Task 1 fixed A (15-epoch released; leak-free resume 16→25); E1–E4 grid on leak-free val; E5 Adam/SGD/cosine (8 epochs); E6 seeds 42/43/44; per-image preds + stats; E5 AdamW/RMSprop cut (D13) |
| 9 Report + rubric + final QA | IN PROGRESS | `report/report.tex`, `report/slides_rubric.tex`, `RUBRIC_MAP.md`; compile + page inspection after E6 |

## Environment
- `tf.config.list_physical_devices()` = CPU only (`PhysicalDevice CPU:0`).
  Fallback per brief: IMG_SIZE 224, BATCH 32, seed 42 (D1).
- Dataset on disk: released train 2647 / val 755 / test 382;
  leak-free train 2632 / val 760 / test 392.
- Toolchain: venv (`venv/`), `bin/tectonic` 0.17.0, PyMuPDF for page inspection.

## Reproducibility
- Every run writes `outputs/logs/<run>.csv`, `<run>_meta.json`, and best/final
  `.keras` checkpoints. `outputs/logs/` and `*.keras` are gitignored (large);
  metrics, tables and figures are committed.
- `train.py` skips a run when its meta+best checkpoint exist; use `--force`
  only for an intentional rerun.
- Experiment commands: `scripts/run_b_grid.sh` (E1–E4), `scripts/run_e5.sh`
  (E5), and the Task-1/E6 commands documented in `RUBRIC_MAP.md`.
