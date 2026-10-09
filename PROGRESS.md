# PROGRESS.md — phase tracker (updated after every phase)

| Phase | Status | Key numbers / notes |
|---|---|---|
| 0 Recon + setup | DONE | macOS 25.5 arm64, Py3.12, TF 2.21 CPU-only, Tectonic 0.17.0; venv + requirements installed; dataset staged at `Dataset/` |
| 1 Dataset audit | TODO | — |
| 2 Implementation + smoke test | TODO | — |
| 3 Experiments | TODO | — |
| 4 Evaluation + analysis | TODO | — |
| 5 Research survey | TODO | — |
| 6 Beamer slides | TODO | — |
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
