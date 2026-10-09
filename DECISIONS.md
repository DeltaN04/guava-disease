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
