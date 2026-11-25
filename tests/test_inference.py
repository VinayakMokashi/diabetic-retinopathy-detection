from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.test_preprocessing import synthetic_fundus

MODEL = Path(__file__).resolve().parents[1] / "models" / "dr_effnet_b0.pt"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")


@pytest.fixture(scope="module")
def predictor():
    from dr.inference import Predictor
    return Predictor(MODEL)


def test_prediction_is_well_formed(predictor):
    p = predictor.predict(Image.fromarray(synthetic_fundus()))
    assert 0 <= p.grade <= 4
    assert len(p.probabilities) == 5 and sum(p.probabilities) == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= p.severity_score <= 4.0
    assert p.referable == (p.grade >= 2)
    assert p.preprocessed.size == (predictor.size, predictor.size)


def test_tta_and_plain_agree_roughly(predictor):
    img = Image.fromarray(synthetic_fundus())
    a, b = predictor.predict(img, tta=True), predictor.predict(img, tta=False)
    assert abs(a.severity_score - b.severity_score) < 1.0


def test_gradcam_heatmap(predictor):
    overlay_img, heat = predictor.explain(Image.fromarray(synthetic_fundus()))
    assert heat.shape == (predictor.size, predictor.size)
    assert np.isfinite(heat).all() and heat.min() >= 0 and heat.max() <= 1
    assert overlay_img.size == (predictor.size, predictor.size)
