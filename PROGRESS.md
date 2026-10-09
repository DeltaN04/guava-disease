# PROGRESS.md — phase tracker (updated after every phase)

| Phase | Status | Key numbers / notes |
|---|---|---|
| 0 Recon + setup | DONE | macOS 25.5 arm64, Py3.12, TF 2.21 CPU-only, Tectonic 0.17.0; venv + requirements installed; dataset staged at `Dataset/` |
| 1 Dataset audit | DONE | 3784 files, 0 corrupt, all 512x512 RGB PNG; val=755 on disk vs 775 expected (discrepancy reported); **leakage: 100% of val+test share an original with train (454/473 originals span splits)**; `Dataset_leakfree/` built; figs in `outputs/audit/` |
| 2 Implementation + smoke test | DONE (bugfix 9 Oct) | custom CNN + MobileNetV2 transfer (two-phase) implemented; **train/val skew bug found+fixed** (`value_range=(0,1)`, see D8); post-fix smoke (64 imgs, 2 epochs): custom train acc 0.53-0.56, transfer val_acc 0.83; save/reload parity True; fine-tune OK (1839427/2586691 trainable) |
| 3 Experiments | IN PROGRESS (restarted post-D9) | D8 run diverged (train 0.91/val 0.245 BN collapse, killed ep 4, archived `models_buggy/custom_base_bn099.*`); D9 fix verified: `custom_bn09_test` 5 epochs best val **0.9126** (train 0.917); full `--model both` relaunched post-fix (custom 40 + transfer 15+20, background sh_121dd3402001xOExDYplWverso) |
| 4 Evaluation + analysis | TODO (partial) | `evaluate.py` ran on buggy ckpt (test 0.4450/macroF1 0.285/AUC 0.656, cm collapse documented); also crashed on missing `jinja2` (fixed in requirements); full rerun blocked on Phase 3 checkpoints |
| 5 Research survey | IN PROGRESS | 12 core refs + 5 guava leads verified live (Mostafa 2021, Doutoum 2023, Nandi 2022, Kilci/Koklu 2024 verified; Pratama 2025 DROPPED unverifiable); expanding `survey.md` table to 12-13 disease papers + foundational |
| 6 Beamer slides | IN PROGRESS | skeleton `report/slides.tex` (Tectonic-safe, conditional figures); audit figs included, results slots filled after Phase 4 |
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

