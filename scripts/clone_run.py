"""Clone a completed run to a new tag without retraining (seed-42 reuse).

Copies the best/final checkpoints, per-epoch CSV and metadata, and rewrites
the ``run``/``task``/``note`` fields in the cloned metadata. Used only when a
final configuration is identical to an optimisation-grid run; the original
run remains untouched.

Usage:
    ./venv/bin/python scripts/clone_run.py --src custom_e5-adam \
        --dst custom_final-s42 --task E6 --note "reused E5 winner seed 42"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main() -> None:
    """Copy checkpoint/log/meta and update the clone's identifiers."""
    ap = argparse.ArgumentParser(description="Clone a completed run.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--note", default="cloned seed-42 run")
    args = ap.parse_args()

    pairs = [
        (config.MODELS_DIR / f"{args.src}.keras",
         config.MODELS_DIR / f"{args.dst}.keras"),
        (config.MODELS_DIR / f"{args.src}_final.keras",
         config.MODELS_DIR / f"{args.dst}_final.keras"),
        (config.LOGS_DIR / f"{args.src}.csv",
         config.LOGS_DIR / f"{args.dst}.csv"),
        (config.LOGS_DIR / f"{args.src}_meta.json",
         config.LOGS_DIR / f"{args.dst}_meta.json"),
    ]
    for src, dst in pairs:
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
    meta = json.loads((config.LOGS_DIR / f"{args.dst}_meta.json").read_text())
    meta["run"] = args.dst
    meta["task"] = args.task
    meta["note"] = args.note
    meta["cloned_from"] = args.src
    (config.LOGS_DIR / f"{args.dst}_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(f"Cloned {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
