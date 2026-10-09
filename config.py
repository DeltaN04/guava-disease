"""Project-wide configuration for the Guava Fruit Disease Classification project.

All paths are anchored at the repository root (the directory containing this
file). Class names are *derived from disk* at runtime via
:func:`get_class_names` -- never hard-coded -- because the released GFDD24
folders use the names ``Anthracnose``, ``fruit_fly`` and ``healthy_guava``.

Key documented deviations from the original brief (see DECISIONS.md):
  * ``IMG_SIZE`` defaults to 224x224 instead of the native 512x512 because the
    build machine is CPU-only (no NVIDIA GPU on Apple Silicon).
  * ``BATCH_SIZE`` is 32 instead of 16 for the same reason (CPU throughput).
  * Experiments use a single seed (42); multi-seed mean +/- std was skipped
    for time reasons and this is stated wherever results are reported.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths -----
BASE_DIR: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = BASE_DIR / "Dataset"
LEAKFREE_DATASET_DIR: Path = BASE_DIR / "Dataset_leakfree"

OUTPUTS_DIR: Path = BASE_DIR / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
AUDIT_DIR: Path = OUTPUTS_DIR / "audit"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"

for _d in (MODELS_DIR, LOGS_DIR, PLOTS_DIR, AUDIT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------- reproducibility ----
SEED: int = 42

# ------------------------------------------------------------------ data ----
IMG_SIZE: tuple[int, int] = (224, 224)  # CPU fallback; native size is 512x512
BATCH_SIZE: int = 32
NUM_CLASSES: int = 3

# Expected totals from the GFDD24 release notes (train / val / test).
EXPECTED_SPLIT_TOTALS: dict[str, int] = {"train": 2647, "val": 775, "test": 382}

# Human-friendly labels for plots / slides (keys are disk folder names).
DISPLAY_NAMES: dict[str, str] = {
    "Anthracnose": "Anthracnose",
    "fruit_fly": "Fruit Flies",
    "healthy_guava": "Healthy",
}

# --------------------------------------------------------------- training ---
CUSTOM_LR: float = 1e-3
CUSTOM_MAX_EPOCHS: int = 40

HEAD_LR: float = 1e-3
HEAD_EPOCHS: int = 15
FT_LR: float = 1e-5
FT_EPOCHS: int = 20
FT_UNFREEZE_LAST: int = 30
BACKBONE_NAME: str = "MobileNetV2"

ES_PATIENCE: int = 8
RLROP_PATIENCE: int = 4
RLROP_FACTOR: float = 0.5


def get_class_names(
    dataset_dir: Path = DATASET_DIR, split: str = "train"
) -> list[str]:
    """Return sorted class (folder) names for a split, read from disk.

    Sorting matches ``tf.keras.utils.image_dataset_from_directory``, which
    orders ``class_names`` alphanumerically, so label indices agree.
    """
    split_dir = Path(dataset_dir) / split
    names = sorted(
        p.name for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not names:
        raise FileNotFoundError(f"No class folders found in {split_dir}")
    return names
