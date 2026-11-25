"""Data figures for the README: grade distribution per split, and raw vs preprocessed examples per grade.

Usage:
    python scripts/make_data_figures.py --source "<...>/Dataset Full/data"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dr import GRADES  # noqa: E402
from dr.preprocessing import preprocess_fundus  # noqa: E402

FOLDERS = {"train": "training", "val": "validation", "test": "test"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    p.add_argument("--out", type=Path, default=Path("reports/figures"))
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.splits)

    counts = df.pivot_table(index="grade", columns="split", values="image", aggfunc="count")[["train", "val", "test"]]
    ax = counts.plot.bar(figsize=(8, 4), rot=0, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_xticklabels([f"{g}\n{GRADES[g]}" for g in counts.index])
    ax.set_xlabel("")
    ax.set_ylabel("images")
    ax.set_title("Grade distribution by split (patient-level split, 2,111 images)")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out / "class_distribution.png", dpi=150)
    plt.close()

    rng = np.random.default_rng(args.seed)
    train = df[df.split == "train"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
    for g in range(5):
        row = train[train.grade == g].iloc[int(rng.integers(0, (train.grade == g).sum()))]
        with Image.open(args.source / FOLDERS[row.original_split] / str(g) / f"{row.image}.jpeg") as im:
            raw = im.convert("RGB")
        axes[0, g].imshow(raw)
        axes[0, g].set_title(f"grade {g}: {GRADES[g]}", fontsize=10)
        axes[1, g].imshow(preprocess_fundus(raw, size=384))
        for a in axes[:, g]:
            a.axis("off")
    axes[0, 0].text(-0.08, 0.5, "original", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.08, 0.5, "model input", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle("Preprocessing: crop, resize to 384 px, subtract the local average colour, mask outside the disc")
    fig.tight_layout()
    fig.savefig(args.out / "preprocessing_examples.png", dpi=110)
    plt.close(fig)
    print("wrote", args.out / "class_distribution.png", "and", args.out / "preprocessing_examples.png")


if __name__ == "__main__":
    main()
