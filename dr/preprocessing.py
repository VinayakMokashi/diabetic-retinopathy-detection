"""Fundus photograph preprocessing.

Fundus cameras differ in exposure, colour balance and field of view. The pipeline below removes most of that
variation before the network sees the image:

1. crop away the black border around the circular fundus;
2. pad to a square and resize;
3. subtract the local average colour (Ben Graham's preprocessing from the 2015 Kaggle Diabetic Retinopathy
   competition): ``4 * image - 4 * gaussian_blur(image) + 128``. This flattens uneven illumination and makes small
   lesions such as microaneurysms, haemorrhages and exudates stand out against a neutral grey background;
4. mask everything outside the fundus disc to the same grey, which also hides the bright ring the subtraction
   creates at the disc edge.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

GREY = 128


def crop_to_fundus(image: np.ndarray, threshold: float = 15.0) -> np.ndarray:
    """Crop to the bounding box of pixels brighter than ``threshold`` (the fundus disc).

    Returns the input unchanged if almost nothing is brighter than the threshold (e.g. a very dark image).
    """
    grey = image.astype(np.float32).mean(axis=2)
    mask = grey > threshold
    if mask.mean() < 0.05:
        return image
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return image[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


def pad_to_square(image: np.ndarray, fill: int = 0) -> np.ndarray:
    """Pad the shorter side symmetrically so the image becomes square."""
    h, w = image.shape[:2]
    side = max(h, w)
    out = np.full((side, side, image.shape[2]), fill, dtype=image.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    out[top: top + h, left: left + w] = image
    return out


def subtract_local_average(image: np.ndarray, sigma: float) -> np.ndarray:
    """Ben Graham's local average colour subtraction: 4 * image - 4 * blur(image, sigma) + 128."""
    blurred = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32)
    out = 4.0 * image.astype(np.float32) - 4.0 * blurred + GREY
    return np.clip(out, 0, 255).astype(np.uint8)


def circular_mask(image: np.ndarray, radius_fraction: float = 0.92, fill: int = GREY) -> np.ndarray:
    """Set every pixel outside a centred circle (radius = fraction of half the side) to ``fill``."""
    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = radius_fraction * min(h, w) / 2.0
    outside = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
    out = image.copy()
    out[outside] = fill
    return out


def preprocess_fundus(image: Image.Image, size: int = 384, radius_fraction: float = 0.92) -> Image.Image:
    """Full preprocessing: crop, square, resize to ``size`` x ``size``, local-average subtraction, circular mask.

    The blur sigma is 1/30 of the fundus radius, as in Graham's original competition code.
    """
    array = np.asarray(image.convert("RGB"))
    array = pad_to_square(crop_to_fundus(array))
    array = np.asarray(Image.fromarray(array).resize((size, size), Image.Resampling.LANCZOS))
    array = subtract_local_average(array, sigma=(size / 2.0) / 30.0)
    array = circular_mask(array, radius_fraction=radius_fraction)
    return Image.fromarray(array)
