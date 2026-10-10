# PROGRESS.md — phase tracker (updated after every phase)

| Phase | Status | Key numbers / notes |
|---|---|---|
| 0 Recon + setup | DONE | macOS 25.5 arm64, Py3.12, TF 2.21 CPU-only, Tectonic 0.17.0; venv + requirements installed; dataset staged at `Dataset/` |
| 1 Dataset audit | DONE | 3784 files, 0 corrupt, all 512x512 RGB PNG; val=755 on disk vs 775 expected (discrepancy reported); **leakage: 100% of val+test share an original with train (454/473 originals span splits)**; `Dataset_leakfree/` built; figs in `outputs/audit/` |
| 2 Implementation + smoke test | DONE (bugfix 9 Oct) | custom CNN + MobileNetV2 transfer (two-phase) implemented; **train/val skew bug found+fixed** (`value_range=(0,1)`, see D8); post-fix smoke (64 imgs, 2 epochs): custom train acc 0.53-0.56, transfer val_acc 0.83; save/reload parity True; fine-tune OK (1839427/2586691 trainable) |
| 3 Experiments | DONE | A base 9ep val 0.9364 (31.6min), B base 15+20 val 0.9960 (14.5min); frozen 15ep 0.9907, noaug 15+18 0.9974; leakfree: A-partial 17/40 best 0.9737, B 15+17 0.9855. Weights/logs gitignored (large); metrics+tables committed; repro via `train.py` |
| 7 Final QA + handoff | DONE | smoke re-passed post-fix; 0 Beamer warnings; 18/18 pages PNG-inspected; no placeholders; README+analysis+slides committed |
| 4 Evaluation + analysis | DONE (base) | test: A 0.9110/F1 0.9173/AUC 0.9949, B 0.9921/0.9911/0.9999, frozen 0.9948, noaug 0.9948; CMs (A: 26 Anthracnose→FF; B: 3 errs), ROC, loss/acc curves, Grad-CAM panels, `outputs/results/analysis.md` |
| 5 Research survey | DONE | 12-study table (§8) + families (§9) + gap (§10) in `survey.md`; 22-entry `references.bib`; all verified live 9 Oct; Pratama 2025 explicitly dropped (unverifiable) |
| 6 Beamer slides | DONE | 18 frames, Madrid navy/blue, footline, booktabs real table, CM+Grad-CAM, refs, thank-you; Tectonic compiles; page-by-page PNG inspect, 2 table overflows fixed |
| 7 Final QA + handoff | TODO | — |

## Phase 0 detail
- `tf.config.list_physical_devices()` = CPU only (`PhysicalDevice CPU:0`).
  Fallback per brief rule 7: IMG_SIZE 224, BATCH 32, seed 42 (see DECISIONS.md).
- Zip-level counts: train 1080+918+649=2647 (expected 2647 OK);
  val 308+262+185=755 (expected 775, mismatch of 20 — audit to confirm);
  test 156+132+94=382 (expected 382 OK). Total 3784.
- Class folders on disk: `Anthracnose`, `fruit_fly`, `healthy_guava`.
- Toolchain: venv (`venv/`), `requirements.txt` installed OK
  (tensorflow 2.21.0, numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2,
  seaborn 0.13.2, scikit-learn 1.9.1, pillow 12.3.0, ImageHash 4.3.2,
  tqdm 4.70.1). Tectonic binary in `bin/`.

## Phase 3 detail (live)
- `custom_base` started 21:57 IST; epoch 1: loss 1.6024, acc 0.3476,
  val_loss 1.5664, val_acc 0.3470, val_auc 0.5874. Checkpoint saved
  (`outputs/models/custom_base.keras`, 14 MB). Pace ~169 s/epoch.
- Plan (sequential): custom_base -> transfer_MobileNetV2_base (15 head + 20 FT)
  -> frozen + noaug + leakfree ablations -> evaluate -> Grad-CAM -> figures.

