"""Render every PDF page to PNG for visual inspection.

Usage:
    ./venv/bin/python scripts/render_pages.py --pdf report/slides_rubric.pdf \
        --out-dir /tmp/slides_rubric_pages
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def main() -> None:
    """Render pages at a readable DPI and print the page count."""
    ap = argparse.ArgumentParser(description="Render PDF pages to PNG.")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=args.dpi)
        path = out / f"page-{i + 1:02d}.png"
        pix.save(path)
        print(path)
    print(f"pages={len(doc)}")


if __name__ == "__main__":
    main()
