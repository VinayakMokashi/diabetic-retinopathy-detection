"""Metrics for ordinal DR grading: quadratic weighted kappa, threshold decoding, referable-DR detection and
patient-level bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score,
                             roc_auc_score)

from . import NUM_CLASSES, REFERABLE_GRADE

DEFAULT_THRESHOLDS = (0.5, 1.5, 2.5, 3.5)


def quadratic_kappa(y_true, y_pred) -> float:
    """Cohen's kappa with quadratic weights: the standard metric for DR grading (and the Kaggle 2015 metric)."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(NUM_CLASSES))))


def expected_grade(probs: np.ndarray) -> np.ndarray:
    """Probability-weighted mean grade, a continuous score in [0, 4]."""
    return np.asarray(probs) @ np.arange(probs.shape[1])


def decode(scores: np.ndarray, thresholds=DEFAULT_THRESHOLDS) -> np.ndarray:
    """Map continuous grade scores to integer grades using four increasing cut-points."""
    return np.digitize(np.asarray(scores), np.sort(np.asarray(thresholds))).astype(int)


def optimize_thresholds(scores: np.ndarray, y_true: np.ndarray, init=DEFAULT_THRESHOLDS, n_rounds: int = 3,
                        grid: np.ndarray | None = None) -> tuple[float, ...]:
    """Tune the four cut-points to maximise quadratic kappa (coordinate ascent over a grid).

    Run this on validation predictions only; applying the tuned cut-points to the same data would overstate kappa.
    """
    grid = np.linspace(0.0, 4.0, 161) if grid is None else grid
    t = list(init)
    best = quadratic_kappa(y_true, decode(scores, t))
    for _ in range(n_rounds):
        improved = False
        for i in range(len(t)):
            lo = t[i - 1] if i > 0 else 0.0
            hi = t[i + 1] if i < len(t) - 1 else 4.0
            for candidate in grid[(grid > lo) & (grid < hi)]:
                trial = t.copy()
                trial[i] = float(candidate)
                k = quadratic_kappa(y_true, decode(scores, trial))
                if k > best + 1e-9:
                    best, t, improved = k, trial, True
        if not improved:
            break
    return tuple(round(v, 4) for v in t)


def referable_probability(probs: np.ndarray) -> np.ndarray:
    """P(grade >= 2) from the five class probabilities."""
    return np.asarray(probs)[:, REFERABLE_GRADE:].sum(axis=1)


def summarize(y_true: np.ndarray, probs: np.ndarray, thresholds=DEFAULT_THRESHOLDS) -> dict:
    """All headline metrics for one set of predictions."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    pred = decode(expected_grade(probs), thresholds)
    ref_true = y_true >= REFERABLE_GRADE
    ref_pred = pred >= REFERABLE_GRADE
    any_true = y_true >= 1
    return {
        "n_images": int(len(y_true)),
        "quadratic_kappa": quadratic_kappa(y_true, pred),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
        "within_one_grade": float(np.mean(np.abs(pred - y_true) <= 1)),
        "per_grade_recall": [float(v) for v in recall_score(y_true, pred, labels=list(range(NUM_CLASSES)),
                                                            average=None, zero_division=0)],
        "referable_auc": float(roc_auc_score(ref_true, referable_probability(probs))),
        "referable_sensitivity": float(ref_pred[ref_true].mean()),
        "referable_specificity": float((~ref_pred[~ref_true]).mean()),
        "any_dr_auc": float(roc_auc_score(any_true, 1.0 - probs[:, 0])),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(NUM_CLASSES))).tolist(),
    }


def summarize_hard(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """Metrics for a model that only outputs hard grades (no probabilities)."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    ref_true, ref_pred = y_true >= REFERABLE_GRADE, pred >= REFERABLE_GRADE
    return {
        "n_images": int(len(y_true)),
        "quadratic_kappa": quadratic_kappa(y_true, pred),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
        "within_one_grade": float(np.mean(np.abs(pred - y_true) <= 1)),
        "per_grade_recall": [float(v) for v in recall_score(y_true, pred, labels=list(range(NUM_CLASSES)),
                                                            average=None, zero_division=0)],
        "referable_sensitivity": float(ref_pred[ref_true].mean()),
        "referable_specificity": float((~ref_pred[~ref_true]).mean()),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(NUM_CLASSES))).tolist(),
    }


def patient_bootstrap_ci(metric: Callable[[np.ndarray], float], groups: np.ndarray, n_boot: int = 2000,
                         seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of ``metric(indices)`` resampling whole patients (both eyes stay together).

    Both eyes of one patient are strongly correlated, so resampling images independently would make the interval
    too narrow.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[g] for g in picked])
        try:
            values.append(metric(idx))
        except ValueError:  # e.g. a resample with only one class for an AUC
            continue
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
