"""Datasets, augmentations and test-time augmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from .preprocessing import GREY


class FundusDataset(torch.utils.data.Dataset):
    """Preprocessed fundus images (``<image_dir>/<image>.jpg``) with their grades."""

    def __init__(self, frame: pd.DataFrame, image_dir: str | Path, transform):
        self.paths = [Path(image_dir) / f"{name}.jpg" for name in frame["image"]]
        self.labels = frame["grade"].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(self.labels[index])


def train_transform(mean, std) -> T.Compose:
    """Augmentations that keep a fundus photograph plausible.

    The retina has no canonical orientation in these images (left and right eyes, cameras held at different
    angles), so any rotation and both flips are fair. Colour jitter is mild because colour carries diagnostic
    information (exudates are yellow, haemorrhages red).
    """
    return T.Compose([
        T.RandomAffine(degrees=180, scale=(0.95, 1.05), fill=GREY),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transform(mean, std) -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def channel_stats(paths: list[Path], max_images: int = 500, seed: int = 0) -> tuple[list[float], list[float]]:
    """Per-channel mean and standard deviation (in [0, 1] units) over a sample of preprocessed training images."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(paths), size=min(max_images, len(paths)), replace=False)
    total, total_sq, count = np.zeros(3), np.zeros(3), 0
    for i in chosen:
        with Image.open(paths[i]) as image:
            a = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        total += a.reshape(-1, 3).sum(axis=0)
        total_sq += (a.reshape(-1, 3) ** 2).sum(axis=0)
        count += a.shape[0] * a.shape[1]
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    return [round(float(v), 4) for v in mean], [round(float(v), 4) for v in std]


@torch.no_grad()
def predict_proba(model: torch.nn.Module, x: torch.Tensor, tta: bool = True) -> torch.Tensor:
    """Softmax probabilities, averaged over the four flips of the image when ``tta`` is on."""
    views = [x]
    if tta:
        views += [torch.flip(x, dims=[3]), torch.flip(x, dims=[2]), torch.flip(x, dims=[2, 3])]
    probs = [torch.softmax(model(v), dim=1) for v in views]
    return torch.stack(probs).mean(dim=0)
