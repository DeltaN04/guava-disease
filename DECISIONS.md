# DECISIONS.md — running record of consequential choices

## D0. Dataset actually found (Phase 0)
- The `~/Downloads/student_resource/dataset/` TSV set belongs to a different
  course project (text data), NOT guava. The real data is
  `Guava Fruit Disease Dataset.zip` (1.63 GB) in `~/Downloads`, containing
  `GuavaDiseaseDataset.zip` with `train/ val/ test/` x
  `{Anthracnose, fruit_fly, healthy_guava}` = 3,784 image files.
- Staged a copy at `guava-disease/Dataset/` (project-local, reproducible).
  Original zips untouched in `~/Downloads`.

## D1. CPU-only execution (Phase 0)
- Machine: MacBook Air, Apple Silicon, macOS 25.5. `nvidia-smi` absent;
  `tf.config.list_physical_devices()` reports CPU only. No GPU available.
- Consequence (brief allows graceful fallback, rule 7): default
  `IMG_SIZE = (224, 224)` instead of 512x512, `BATCH_SIZE = 32` instead of 16.
  Rationale: MobileNetV2/ImageNet weights are native 224; full 512px training
  of both models on CPU would take many hours and risk OOM from the custom
  CNN's dense feature maps. Documented in `config.py`, README and slides.

## D2. Single seed (Phase 3 scope cut)
- 3-seed mean +/- std would triple CPU training time. We run seed 42 once
  and state this plainly in results, report and slides (brief permits this).

## D3. Tectonic instead of latexmk (Phase 0)
- No LaTeX distribution installed and no Homebrew / no-password sudo.
  Downloaded the official Tectonic 0.17.0 `aarch64-apple-darwin` binary
  (self-contained TeX engine) into `guava-disease/bin/`. Slides compile with
  `bin/tectonic`. Beamer + booktabs fetched from TeXLive bundles on demand
  (network available).

## D4. Validation split count mismatch (Phase 1, to verify in audit)
- Zip listing shows val = 308 + 262 + 185 = 755 files, but the release notes
  say 775. Train (2647) and test (382) match. Audit will confirm on disk and
  the discrepancy is reported, not hidden.

## D5. Class names read from disk (no hardcoding)
- Disk folders: `Anthracnose`, `fruit_fly`, `healthy_guava`. Alphabetical
  order (as used by `image_dataset_from_directory`) is
  `[Anthracnose, fruit_fly, healthy_guava]`; this ordering defines label
  indices everywhere. Display names for plots/slides: Anthracnose,
  Fruit Flies, Healthy.

## D6. Audit cluster-key bugfix: timestamp filenames (Phase 1)
- Two filename patterns exist: `<int>_unsharp_clahe_augmented_N.png` (all
  classes) and `<YYYYMMDD>_<HHMMSS>_unsharp_clahe_augmented_N.png` (600
  fruit-fly files). First audit version keyed clusters on the leading int,
  which lumps all same-day timestamp files into one bogus cluster.
  Fixed `orig_id_of()` to use the full `YYYYMMDD_HHMMSS` stamp for those.
  Verified: 473 unique (class, key) clusters x exactly 8 variants each =
  3,784 files, matching the "473 originals x8" release story exactly.

## D7. Leakage headline (Phase 1, verified on disk)
- 454 of 473 originals (96%) have augmented variants scattered across
  multiple splits. **100% of val (755/755) and 100% of test (382/382)
  images share their source original with the training set** (spot-checked
  by hand, e.g. test Anthracnose `101_..._augmented_5.png` has 6 same-
  original siblings in train). phash near-duplicate search (threshold 6)
  only flags 8/5/3 cross-split pairs — augmentation perturbs hashes, so
  the filename-prefix analysis is the definitive evidence and both numbers
  are reported. Consequence: released-split test accuracy is optimistic;
  we report it AND a `Dataset_leakfree/` re-split (cluster-grouped 70/20/10,
  seed 42) per the brief. Val total is 755 on disk vs 775 in the release
  notes (train 2647 and test 382 match); reported as a discrepancy.

## D8. Train/val skew bugfix: RandomBrightness/Contrast value_range (Phase 2/3)
- Symptom (9 Oct 2026): `custom_base` stuck at val_acc 0.35-0.45 for 10
  epochs, test acc 0.4450 / macroF1 0.285, confusion matrix collapsed
  (148/110/94 predicted Anthracnose, 0 Healthy). Train loss 1.60->1.15
  never beat random-guess 1.099. EarlyStopping (val_loss, patience 8)
  halted at epoch 10.
- Root cause: pipeline rescales to [0, 1] *before* augmenting, but
  `RandomContrast` / `RandomBrightness` default to `value_range=(0, 255)`
  in TF 2.21. Brightness delta = factor x 255 ~= 25.5, so train batches
  spanned [0, ~24] while val/test stayed [0, 1]. Verified live:
  `rescaled 0.0-1.0 -> augmented 0.0-23.63`, culprit isolated to
  `RandomBrightness`. Fix: `value_range=(0, 1)` on both layers in
  `data_pipeline.augmentation_layers()` (see docstring there).
- Consequence: all pre-fix checkpoints/logs (`custom_base` 10-epoch run)
  are INVALID (train/val distribution mismatch). Moved to
  `outputs/models_buggy/` for provenance, deleted from `outputs/models/`
  and `outputs/logs/`, and restarted training from scratch. Smoke after
  fix (64 imgs, 2 epochs): custom train acc 0.53-0.56 (was 0.34),
  transfer val_acc 0.83 — learning restored. Full `train.py --model both`
  relaunched 9 Oct ~23:20 IST (background, CPU, ~4-5 h expected).
- Also added missing `jinja2` to `requirements.txt` (pandas `to_latex`
  via Styler needs it; `evaluate.py` crashed without it).

## D9. BatchNorm momentum fix: inference collapse (Phase 2/3, 10 Oct 2026)
- Symptom: post-D8 custom run diverged — train acc 0.823->0.909 over
  3 epochs while val_acc frozen at 0.245 (Healthy share) and val_loss
  3.95->9.79 exploding. Killed at epoch 4 (PID 38300).
- Diagnosis (live, on `custom_base.keras`): in *training* mode
  (batch stats) train acc 0.854 balanced across classes, but in
  *inference* mode (moving averages) the same batches predicted ALL
  class 2 (healthy) with softmax ~[0.001, 0.033, 0.966], train acc 0.29
  / val acc 0.0. BN8 inference mean +0.24 / std 0.89 vs training-mode
  0.00 / 0.99 — moving averages lagged fast-shifting batch stats (model
  hits 0.82 train in 1 epoch with Adam 1e-3; momentum 0.99 chases a
  moving target). Moving stats themselves looked sane (means 0-0.8,
  vars 0.4-6), just stale.
- Fix (spec-compliant — spec fixes architecture/LR/dropout but not BN
  momentum): `momentum=0.9` on all 9 custom-CNN BatchNorm layers in
  `models.py:_conv_block()` + head BN; distinct per-layer augmentation
  seeds (seed+i) in `data_pipeline.py` for independent yet reproducible
  draws. Diverged ckpt archived to `outputs/models_buggy/custom_base_bn099.*`.
- Verified: `train.py --model custom --epochs-custom 5 --tag bn09_test`
  → best val_acc **0.9126** (ep 4, train 0.917/val_loss 0.237),
  train/val track together, save/reload parity True. Full
  `train.py --model both` relaunched post-fix (background).

## D10. Overnight leakfree-custom death + partial control (Phase 3, 10 Oct)
- The queued `train.py --model both --data-dir Dataset_leakfree --tag
  leakfree` died silently at custom epoch 17/40 (no traceback; likely Mac
  sleep; no `_meta.json` written). Best ckpt `custom_leakfree.keras`
  (ep 15, val 0.9737) survives and is usable.
- Given the 2-hour morning deadline, scope: evaluate the surviving
  custom-leakfree ckpt as a *partial* control (stated as interrupted
  17/40, longer-trained than released-A so not strictly comparable),
  train transfer-leakfree fully (15+20, ~15 min, fast), and report both
  plainly in `outputs/results/analysis.md` and slides. No re-run of the
  3.7h custom-leakfree job.

## D11. Final-experiments branch + corrected early stopping (25 Oct 2026)
- Created branch `final-experiments` from `main` (HEAD `d72a501`).
- `train.py` now defaults EarlyStopping to `val_accuracy` (patience 8,
  `restore_best_weights=True`) and exposes `--es-monitor`, `--es-patience`,
  `--start-from-epoch`. The fixed Model A runs use `val_accuracy`; the old
  `custom_base` run (val_loss, stopped at epoch 9) is retained as provenance
  only.
- Released-split fixed Model A is capped at 15 epochs (the task requires
  "at least 15"); the leak-free run resumes the surviving 16-epoch checkpoint
  and continues to 40 epochs (cut documented in D14 if the CPU budget forces
  an earlier stop).
- Added per-run metadata fields (`task`, `note`, `optimizer`, LRs, schedule,
  unfreeze depth, data split, best epoch) and resumability: a run whose
  `_meta.json` and best checkpoint both exist is skipped unless `--force`.

## D12. Optimisation protocol (25 Oct 2026)
- All E1-E5 tuning uses `Dataset_leakfree/`: train fits, val selects. The
  leak-free test split is touched only by Task 1 (required fixed Model A) and
  by E6 final configurations. Released test is evaluated once per final run to
  quantify leakage, using `evaluate.py --suffix released` so the leak-free and
  released metrics do not overwrite each other.
- E1 uses Adam / AdamW / SGD(Nesterov 0.9) / RMSprop. AdamW uses Keras'
  default weight decay 0.004. SGD uses the task-specified 1e-2 head / 1e-3
  fine-tune. All other optimisers use 1e-3 head / 1e-5 fine-tune.
- E2 is one-factor-at-a-time: head {1e-2, 1e-3, 3e-4}, fine-tune
  {1e-4, 1e-5, 1e-6}. The 1e-3 / 1e-5 reference is the E1 Adam run and is not
  duplicated.
- E3 unfreezes last {0, 30, 60, all} non-BN layers; last-30 is the E1 Adam
  reference. E4 compares MobileNetV2 / EfficientNetB0 / ResNet50 under the
  same two-phase protocol; MobileNetV2 is the E1 Adam reference.
- EfficientNetB0 preprocessing was corrected: Keras EfficientNet contains its
  own Rescaling(1/255)+Normalization, so the model must receive [0,255];
  the previous [-1,1] rescaling would have made every input negative. ResNet50
  uses an RGB-order mean-subtraction approximation (no BGR flip) and this is
  stated wherever its number appears.

## D13. E5 scope cut for Model A (25 Oct 2026)
- Model A is ~3.5-5 CPU-min/epoch at 224px. E5 runs use the 12-epoch cap but
  are executed for **8 epochs** because the two concurrent training streams
  roughly doubled per-epoch wall time; the 8-epoch choice is recorded here,
  not silent. We run Adam + ReduceLROnPlateau, SGD + Nesterov +
  ReduceLROnPlateau, and Adam + cosine decay. AdamW and RMSprop for Model A
  are **not run** because each additional run costs ~30-50 CPU minutes; the
  cut is explicit.
- E6 trains the best E5 Model A setting and the best E1-E4 Model B setting
  with seeds 42/43/44. Seed-42 runs are reused (cloned) when the configuration
  is identical; no duplicate training. E6 Model A uses the same 8-epoch
  budget as E5 so the comparison is consistent.
- The leak-free Task-1 Model A run resumes the 16-epoch survivor and is
  continued to **25 epochs** (9 additional epochs) rather than 40; the old run
  is reported as interrupted 17/40, the resumed run as 25/40, and the
  remaining 15 epochs are an explicit CPU-budget cut. The released fixed run
  is 15 epochs (the task's minimum).

## D14. Provenance incident: smoke test clobbered base artifacts (25 Oct 2026)
- A final smoke command used the default `--tag base`, overwriting
  `custom_base.keras`, `transfer_MobileNetV2_base.keras`, their CSV logs and
  meta files with 1-epoch/64-image smoke artifacts. The *metrics JSONs* from
  the original real runs survive and are still valid evidence for the old
  provenance numbers (A 0.9110, B 0.9921), but per-image predictions for the
  old base runs are gone.
- Consequence: the old base runs are reported as provenance only (no McNemar
  against them). The final A/B comparison uses newly trained fixed/final runs
  with saved per-image predictions. `train.py` now skips a run when its
  meta+checkpoint exist, which prevents this class of accident; smoke tests
  must use a unique tag (e.g. `qa1`).

## D15. Pratama et al. 2025 re-verified (25 Oct 2026)
- The previously dropped JESSI paper was re-checked against the Crossref API:
  DOI `10.59562/jessi.v6i3.10153` returns status 200, title "Transfer
  Learning-Based CNN for Guava Fruit Disease Detection and Classification",
  authors Pratama / Lamada / Surianto, pages 464-477, published 2025-10-21.
  Volume/issue are encoded in the DOI (`v6i3`). Restored to
  `references.bib` and the survey table; not used as a numeric baseline
  because we did not reproduce its split.

## D16. Statistics (25 Oct 2026)
- Per-image predictions saved for every final run. Bootstrap CIs use 2,000
  resamples with seed 42. McNemar uses the exact two-sided binomial test;
  the chi-square-with-continuity statistic is also stored. Differences that
  are not significant at alpha=0.05 are described as within noise; a
  significant difference spanning <=2 test images is flagged separately.
