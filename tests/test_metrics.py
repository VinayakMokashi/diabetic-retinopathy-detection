import numpy as np
import pytest

from dr.metrics import (decode, expected_grade, optimize_thresholds, patient_bootstrap_ci, quadratic_kappa,
                        referable_probability, summarize)


def test_quadratic_kappa_extremes():
    y = np.array([0, 1, 2, 3, 4] * 4)
    assert quadratic_kappa(y, y) == pytest.approx(1.0)
    assert quadratic_kappa(y, 4 - y) < 0


def test_decode_default_cut_points():
    assert decode(np.array([0.0, 0.49, 0.5, 1.6, 2.5, 3.9, 4.0])).tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_expected_grade_and_referable_probability():
    probs = np.array([[1, 0, 0, 0, 0], [0, 0, 0.5, 0, 0.5]], dtype=float)
    assert expected_grade(probs).tolist() == [0.0, 3.0]
    assert referable_probability(probs).tolist() == [0.0, 1.0]


def test_optimize_thresholds_never_worse_and_sorted():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    scores = np.clip(0.6 * y + 0.4 + rng.normal(0, 0.4, 400), 0, 4)   # biased scores: default cut-points are off
    tuned = optimize_thresholds(scores, y)
    assert list(tuned) == sorted(tuned)
    assert quadratic_kappa(y, decode(scores, tuned)) >= quadratic_kappa(y, decode(scores))


def test_summarize_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 0, 2, 4])
    probs = np.eye(5)[y]
    m = summarize(y, probs)
    assert m["quadratic_kappa"] == pytest.approx(1.0)
    assert m["accuracy"] == 1.0 and m["referable_auc"] == 1.0
    assert m["referable_sensitivity"] == 1.0 and m["referable_specificity"] == 1.0


def test_patient_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    groups = np.repeat(np.arange(100), 2)
    lo, hi = patient_bootstrap_ci(lambda i: float(y[i].mean()), groups, n_boot=300)
    assert lo <= y.mean() <= hi
