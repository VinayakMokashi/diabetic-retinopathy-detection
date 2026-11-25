"""Build ``data/splits.csv``: one row per image with its patient, eye, grade and a patient-level split.

The image subset ships pre-split into ``training/``, ``validation/`` and ``test/`` folders, but that split is by
image, not by patient: 49 patients have one eye in training and the other eye in validation or test. Left and right
eyes of the same person usually have the same grade and look alike, so that leaks information. This script keeps
the original assignment wherever possible and moves every image of a shared patient into a single split
(training wins over validation, validation over test), so no patient appears in two splits.

Usage:
    python scripts/make_splits.py --source "<...>/Dataset Full/data" --out data/splits.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ORIGINAL = {"training": "train", "validation": "val", "test": "test"}
PRIORITY = ["train", "val", "test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="folder with training/ validation/ test/")
    parser.add_argument("--out", default=Path("data/splits.csv"), type=Path)
    args = parser.parse_args()

    rows = []
    for folder, split in ORIGINAL.items():
        for grade_dir in sorted((args.source / folder).iterdir()):
            if not grade_dir.is_dir():
                continue
            for path in sorted(grade_dir.glob("*.jpeg")):
                patient, eye = path.stem.split("_")
                rows.append({"image": path.stem, "patient": int(patient), "eye": eye, "grade": int(grade_dir.name),
                             "original_split": split})
    df = pd.DataFrame(rows)

    shared = df.groupby("patient")["original_split"].nunique()
    print(f"{len(df)} images from {df.patient.nunique()} patients; "
          f"{(shared > 1).sum()} patients appear in more than one original split")

    first = df.groupby("patient")["original_split"].agg(lambda s: min(s, key=PRIORITY.index))
    df["split"] = df["patient"].map(first)
    moved = df[df.split != df.original_split]
    print(f"moved {len(moved)} images:", moved.groupby(["original_split", "split"]).size().to_dict())

    assert (df.groupby("patient")["split"].nunique() == 1).all()
    df = df.sort_values(["split", "grade", "patient", "eye"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["image", "patient", "eye", "grade", "split", "original_split"]].to_csv(args.out, index=False)

    table = df.pivot_table(index="split", columns="grade", values="image", aggfunc="count").loc[PRIORITY]
    table["images"] = table.sum(axis=1)
    table["patients"] = df.groupby("split")["patient"].nunique()
    print(table.to_string())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
