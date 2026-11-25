"""Preprocess every image in ``data/splits.csv`` once and cache it as a JPEG.

Usage:
    python scripts/preprocess_images.py --source "<...>/Dataset Full/data" --size 384 --out data/processed/384
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dr.preprocessing import preprocess_fundus  # noqa: E402

FOLDERS = {"train": "training", "val": "validation", "test": "test"}


def process(row: tuple[str, str, int], source: Path, out: Path, size: int) -> str:
    image, original_split, grade = row
    target = out / f"{image}.jpg"
    if not target.exists():
        src = source / FOLDERS[original_split] / str(grade) / f"{image}.jpeg"
        with Image.open(src) as im:
            preprocess_fundus(im, size=size).save(target, quality=95)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="folder with training/ validation/ test/")
    parser.add_argument("--splits", default=Path("data/splits.csv"), type=Path)
    parser.add_argument("--size", default=384, type=int)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    out = args.out or Path("data/processed") / str(args.size)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.splits)
    rows = list(df[["image", "original_split", "grade"]].itertuples(index=False, name=None))
    work = partial(process, source=args.source, out=out, size=args.size)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(work, rows, chunksize=16), 1):
            if i % 250 == 0 or i == len(rows):
                print(f"{i}/{len(rows)}", flush=True)
    print("cached", len(rows), "images in", out)


if __name__ == "__main__":
    main()
