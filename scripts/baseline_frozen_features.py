"""Baseline without fine-tuning: frozen ImageNet features + multinomial logistic regression.

A linear probe: the backbone stays fixed and only a classifier on its pooled features is trained, on the full training
split, with the regularisation strength chosen on the validation split. Comparing it with the fine-tuned network shows
how much fine-tuning the backbone adds.

Usage:
    python scripts/baseline_frozen_features.py --arch efficientnet_b0.ra_in1k --size 384
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dr.data import FundusDataset, channel_stats, eval_transform  # noqa: E402
from dr.metrics import decode, expected_grade, optimize_thresholds, quadratic_kappa, summarize  # noqa: E402


@torch.no_grad()
def features(backbone, frame, data_dir, mean, std) -> np.ndarray:
    loader = DataLoader(FundusDataset(frame, data_dir, eval_transform(mean, std)), batch_size=32)
    out = []
    for x, _ in loader:
        views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]
        out.append(torch.stack([backbone(v) for v in views]).mean(0))
    return torch.cat(out).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="efficientnet_b0.ra_in1k")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()

    torch.set_num_threads(8)
    data_dir = args.data or Path("data/processed") / str(args.size)
    splits = pd.read_csv(args.splits)
    parts = {s: splits[splits.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    mean, std = channel_stats([data_dir / f"{n}.jpg" for n in parts["train"].image])

    cache = Path("data/features") / f"{args.arch}_{args.size}.npz"
    if cache.exists():
        feats = dict(np.load(cache))
    else:
        backbone = timm.create_model(args.arch, pretrained=True, num_classes=0).eval()
        feats = {s: features(backbone, f, data_dir, mean, std) for s, f in parts.items()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **feats)
    y = {s: f.grade.to_numpy() for s, f in parts.items()}
    print("feature dimension:", feats["train"].shape[1])

    best = None
    for c in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        clf.fit(feats["train"], y["train"])
        k = quadratic_kappa(y["val"], decode(expected_grade(clf.predict_proba(feats["val"]))))
        print(f"C={c:<6} validation kappa {k:.4f}")
        if best is None or k > best[0]:
            best = (k, c, clf)
    _, c, clf = best
    val_probs, test_probs = clf.predict_proba(feats["val"]), clf.predict_proba(feats["test"])
    thresholds = optimize_thresholds(expected_grade(val_probs), y["val"])
    m = summarize(y["test"], test_probs, thresholds)
    print(f"chosen C={c}; thresholds {thresholds}; TEST kappa {m['quadratic_kappa']:.4f}, accuracy {m['accuracy']:.4f}, "
          f"referable AUC {m['referable_auc']:.4f}")

    out = parts["test"][["image", "patient", "grade"]].copy()
    for g in range(5):
        out[f"p{g}"] = test_probs[:, g].round(5)
    out["thresholds"] = json.dumps(list(thresholds))
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "baseline_test_predictions.csv", index=False)
    print("wrote", args.out / "baseline_test_predictions.csv")


if __name__ == "__main__":
    main()
