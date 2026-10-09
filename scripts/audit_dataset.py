"""Phase 1: audit the GFDD24 dataset before any modelling.

Reads class folder names from disk (never hard-coded) and writes to
``outputs/audit/``: per-split/per-class counts, image properties,
corrupt-file check, sample/distribution figures, and cross-split
near-duplicate (leakage) analysis via perceptual hashing + filename
original-id prefixes (e.g. ``94_unsharp_clahe_augmented_3.png``).

Usage:
    ./venv/bin/python scripts/audit_dataset.py [--make-leakfree]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import imagehash

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NEAR_DUP_THRESHOLD = 6  # max phash hamming distance = "near duplicate"
TIMESTAMP_RE = re.compile(r"^(\d{8}_\d{6})_")
INT_RE = re.compile(r"^(\d+)_")


def iter_image_files(split_dir: Path) -> list[Path]:
    """Image files under a split dir, skipping dotfiles like .DS_Store."""
    return sorted(
        p for p in split_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in IMG_EXTS
    )


def orig_id_of(path: Path) -> str:
    """Original-image cluster key.

    Two filename patterns exist (both end ``_unsharp_clahe_augmented_N.png``):
      * ``<int>_unsharp_...`` -> the leading int (e.g. ``94``)
      * ``<YYYYMMDD>_<HHMMSS>_unsharp_...`` (fruit-fly only) -> the full
        ``YYYYMMDD_HHMMSS`` stamp. NB: the bare ``YYYYMMDD`` date alone is
        NOT unique (one shooting day yields ~75 originals), so the time
        component is required. Callers qualify the key with the class name.
    """
    m = TIMESTAMP_RE.match(path.name)
    if m:
        return m.group(1)
    m = INT_RE.match(path.name)
    return m.group(1) if m else path.stem


def scan_split(dataset_dir: Path, split: str, class_names: list[str]) -> pd.DataFrame:
    """Decode every image once; record stats + perceptual hash."""
    rows = []
    for cls in class_names:
        files = iter_image_files(dataset_dir / split / cls)
        for path in tqdm(files, desc=f"scan {split}/{cls}", leave=False):
            rec = {"split": split, "class": cls, "path": str(path),
                   "width": None, "height": None, "mode": None,
                   "format": None, "corrupt": False, "phash": None}
            try:
                with Image.open(path) as im:
                    im.load()  # full decode catches truncated files
                    rec.update(width=im.width, height=im.height,
                               mode=im.mode, format=im.format)
                    rec["phash"] = str(imagehash.phash(im))
            except Exception:  # noqa: BLE001 - any decode failure = corrupt
                rec["corrupt"] = True
            rows.append(rec)
    return pd.DataFrame(rows)


_POPCOUNT16 = np.array([bin(i).count("1") for i in range(65536)],
                     dtype=np.uint8)


def _hamming64(a: np.uint64, B: np.ndarray) -> np.ndarray:
    """Vectorised Hamming distance between scalar hash ``a`` and array ``B``."""
    x = np.bitwise_xor(np.uint64(a), B.astype(np.uint64))
    m = np.uint64(0xFFFF)
    return (_POPCOUNT16[(x & m).astype(np.int64)]
            + _POPCOUNT16[((x >> np.uint64(16)) & m).astype(np.int64)]
            + _POPCOUNT16[((x >> np.uint64(32)) & m).astype(np.int64)]
            + _POPCOUNT16[((x >> np.uint64(48)) & m).astype(np.int64)]
            ).astype(np.int64)


def cross_split_pairs(df: pd.DataFrame, max_dist: int) -> dict:
    """Cross-split near-duplicate search on phash values (vectorised).

    Returns per split-pair ``{'exact': n, 'near': n}`` image-pair counts.
    """
    ok = df[~df["corrupt"] & df["phash"].notna()].copy()
    ok["hint"] = ok["phash"].apply(lambda h: int(h, 16))
    out: dict = {}
    splits = ["train", "val", "test"]
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            A = ok[ok.split == splits[i]]["hint"].to_numpy(dtype=np.uint64)
            B = ok[ok.split == splits[j]]["hint"].to_numpy(dtype=np.uint64)
            exact = near = 0
            for a in A:  # loop over smaller side, vectorised over B
                d = _hamming64(a, B)
                exact += int(np.sum(d == 0))
                near += int(np.sum((d > 0) & (d <= max_dist)))
            out[(splits[i], splits[j])] = {"exact": exact, "near": near}
    return out


def prefix_leakage(df: pd.DataFrame) -> dict:
    """Leakage via filename original-id prefixes (ground truth for GFDD24)."""
    n_prefix_leaked, leaked_val, leaked_test = 0, 0, 0
    grp = df.groupby([df["path"].apply(lambda p: orig_id_of(Path(p))), "class"])
    for (_pid, _cls), g in grp:
        splits = sorted(g["split"].unique())
        if len(splits) > 1:
            n_prefix_leaked += 1
            if "train" in splits:
                leaked_val += int((g["split"] == "val").sum())
                leaked_test += int((g["split"] == "test").sum())
    return {"n_prefixes_across_splits": n_prefix_leaked,
            "val_files_sharing_prefix_with_train": leaked_val,
            "test_files_sharing_prefix_with_train": leaked_test}


def plot_distribution(df: pd.DataFrame, out: Path, class_names: list[str]) -> None:
    """Grouped bar chart of per-split, per-class counts."""
    labels = [config.DISPLAY_NAMES.get(c, c) for c in class_names]
    x = np.arange(len(class_names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, split in enumerate(["train", "val", "test"]):
        counts = [int(((df.split == split) & (df["class"] == c)).sum())
                  for c in class_names]
        bars = ax.bar(x + (k - 1) * 0.25, counts, 0.25, label=split)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Images")
    ax.set_title("GFDD24 class distribution per split (as found on disk)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_samples(dataset_dir: Path, class_names: list[str],
                 out: Path, n: int = 4) -> None:
    """Grid of ``n`` evenly spaced training samples per class."""
    fig, axes = plt.subplots(len(class_names), n,
                             figsize=(3 * n, 3 * len(class_names)))
    for i, cls in enumerate(class_names):
        files = iter_image_files(dataset_dir / "train" / cls)
        idx = np.linspace(0, len(files) - 1, min(n, len(files)), dtype=int)
        for j in range(n):
            ax = axes[i][j] if len(class_names) > 1 else axes[j]
            ax.axis("off")
            if j < len(idx):
                with Image.open(files[idx[j]]) as im:
                    ax.imshow(np.asarray(im.convert("RGB")))
                ax.set_title(
                    f"{config.DISPLAY_NAMES.get(cls, cls)}\n"
                    f"{files[idx[j]].name[:28]}", fontsize=8)
    fig.suptitle("Sample training images per class", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def make_leakfree_split(df: pd.DataFrame) -> Path:
    """Copy images into ``Dataset_leakfree/`` split by original-id cluster.

    Within each class, clusters are shuffled with seed 42 and dealt to
    train/val/test to hit 70/20/10 file ratios as closely as possible.
    """
    out = config.LEAKFREE_DATASET_DIR
    if out.exists():
        shutil.rmtree(out)
    rng = np.random.default_rng(config.SEED)
    manifest = []
    for cls in sorted(df["class"].unique()):
        g = df[(df["class"] == cls) & ~df["corrupt"]]
        clusters: dict = defaultdict(list)
        for _, r in g.iterrows():
            clusters[f"{cls}::{orig_id_of(Path(r['path']))}"].append(r["path"])
        keys = list(clusters)
        rng.shuffle(keys)
        total = sum(len(clusters[k]) for k in keys)
        targets = {"train": total * 0.7, "val": total * 0.2, "test": total * 0.1}
        have = {"train": 0, "val": 0, "test": 0}
        for k in keys:
            nxt = min(("train", "val", "test"),
                      key=lambda s: have[s] / max(targets[s], 1))
            for src in clusters[k]:
                dst = out / nxt / cls / Path(src).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                manifest.append({"src": src, "split": nxt})
            have[nxt] += len(clusters[k])
    pd.DataFrame(manifest).to_csv(config.AUDIT_DIR / "leakfree_manifest.csv",
                                  index=False)
    return out

def md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (avoids a tabulate dependency)."""
    head = "| " + " | ".join([""] + list(df.columns)) + " |"
    sep = "|" + "|".join(["---"] * (len(df.columns) + 1)) + "|"
    body = ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |"
            for i, row in zip(df.index, df.values.tolist())]
    return "\n".join([head, sep] + body)


def main() -> None:
    """Run the full audit and write figures, tables and the summary report."""
    ap = argparse.ArgumentParser(description="Audit the GFDD24 dataset.")
    ap.add_argument("--data-dir", default=str(config.DATASET_DIR))
    ap.add_argument("--make-leakfree", action="store_true")
    args = ap.parse_args()

    dataset_dir = Path(args.data_dir)
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    class_names = config.get_class_names(dataset_dir, "train")
    print(f"Class folders on disk: {class_names}", flush=True)

    df = pd.concat(
        [scan_split(dataset_dir, s, class_names) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    df.to_csv(config.AUDIT_DIR / "file_index.csv", index=False)

    counts = (df.groupby(["split", "class"]).size().unstack(fill_value=0)
                .reindex(class_names, axis=1))
    counts.to_csv(config.AUDIT_DIR / "counts.csv")
    totals = df.groupby("split").size().to_dict()
    print(counts.to_string(), flush=True)
    print(f"Totals: {totals} expected: {config.EXPECTED_SPLIT_TOTALS}", flush=True)

    ok = df[~df["corrupt"]]
    print(f"Corrupt/unreadable files: {int(df['corrupt'].sum())}", flush=True)
    print("Size distribution:", ok.groupby(["width", "height"]).size().to_dict(),
          flush=True)
    print("Modes:", ok["mode"].value_counts().to_dict(), flush=True)
    print("Formats:", ok["format"].value_counts().to_dict(), flush=True)

    plot_distribution(df, config.AUDIT_DIR / "class_distribution.png", class_names)
    plot_samples(dataset_dir, class_names, config.AUDIT_DIR / "samples.png")
    print("Figures written", flush=True)


    pairs = cross_split_pairs(df, NEAR_DUP_THRESHOLD)
    prefix = prefix_leakage(df)
    leak = {"phash_threshold": NEAR_DUP_THRESHOLD,
            "pairs": {f"{a}-{b}": v for (a, b), v in pairs.items()},
            "prefix": prefix}
    with open(config.AUDIT_DIR / "leakage.json", "w") as f:
        json.dump(leak, f, indent=2)
    print("Leakage: " + json.dumps(leak, indent=2), flush=True)

    if args.make_leakfree:
        print("Building leak-free re-split...", flush=True)
        out = make_leakfree_split(df)
        print(f"Leakage-free copy written to {out}", flush=True)

    pair_rows = [{"pair": f"{a}-{b}", **v} for (a, b), v in pairs.items()]
    lines = [
        "# Dataset audit summary (generated -- do not hand-edit)",
        "",
        f"Class folders on disk: {', '.join(class_names)}",
        "",
        "## Per-split, per-class counts",
        "",
        md_table(counts),
        "",
        f"Totals: {totals} -- expected {config.EXPECTED_SPLIT_TOTALS}.",
        "",
        f"Corrupt/unreadable files: {int(df['corrupt'].sum())}",
        "",
        "Image sizes: " + str(ok.groupby(['width', 'height']).size().to_dict()),
        "Modes: " + str(ok['mode'].value_counts().to_dict()),
        "Formats: " + str(ok['format'].value_counts().to_dict()),
        "",
        "## Cross-split near-duplicate (leakage) analysis",
        "",
        f"phash hamming threshold: {NEAR_DUP_THRESHOLD}",
        "",
        md_table(pd.DataFrame(pair_rows).set_index("pair")),
        "",
        f"Filename prefixes spanning splits: {prefix['n_prefixes_across_splits']}",
        "Val files sharing an original with train: "
        f"{prefix['val_files_sharing_prefix_with_train']}",
        "Test files sharing an original with train: "
        f"{prefix['test_files_sharing_prefix_with_train']}",
        "",
    ]
    (config.AUDIT_DIR / "audit_summary.md").write_text("\n".join(lines))
    print(f"Wrote {config.AUDIT_DIR}", flush=True)


if __name__ == "__main__":
    main()

