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
