import numpy as np
from PIL import Image

from dr.preprocessing import GREY, circular_mask, crop_to_fundus, pad_to_square, preprocess_fundus


def synthetic_fundus(h=300, w=420, radius=120, offset=(10, -30)) -> np.ndarray:
    """A reddish disc on a black background, off-centre, like an uncropped fundus photograph."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2 + offset[0], w // 2 + offset[1]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    img[disc] = (180, 90, 40)
    return img


def test_crop_to_fundus_finds_the_disc():
    img = synthetic_fundus()
    cropped = crop_to_fundus(img)
    assert cropped.shape[0] == cropped.shape[1] == 2 * 120 + 1
    assert cropped.mean() > img.mean()


def test_crop_leaves_dark_images_alone():
    dark = np.full((50, 60, 3), 3, dtype=np.uint8)
    assert crop_to_fundus(dark).shape == dark.shape


def test_pad_to_square_centres_the_image():
    img = np.ones((10, 30, 3), dtype=np.uint8) * 7
    out = pad_to_square(img)
    assert out.shape == (30, 30, 3)
    assert (out[10:20] == 7).all() and (out[:10] == 0).all()


def test_circular_mask_greys_the_corners_only():
    img = np.zeros((101, 101, 3), dtype=np.uint8)
    out = circular_mask(img, radius_fraction=0.9)
    assert (out[0, 0] == GREY).all()
    assert (out[50, 50] == 0).all()


def test_preprocess_fundus_shape_and_background():
    out = preprocess_fundus(Image.fromarray(synthetic_fundus()), size=128)
    a = np.asarray(out)
    assert out.size == (128, 128) and a.dtype == np.uint8 and a.shape == (128, 128, 3)
    assert (a[0, 0] == GREY).all()                  # outside the disc is neutral grey
    assert abs(a[64, 64].astype(int) - GREY).max() < 20  # a flat disc becomes near-grey after local-average removal
