"""Evaluate a trained checkpoint on the held-out test split and write metrics, predictions and figures.

Confidence intervals come from a patient-level bootstrap (both eyes of a patient are resampled together).
Optionally scores other models' prediction files (such as the frozen-feature baseline) on the same test images for a
side-by-side comparison.

Usage:
    python scripts/evaluate.py --checkpoint models/dr_effnet_b0.pt \
        --compare "Frozen EfficientNet-B0 features + logistic regression=reports/baseline_test_predictions.csv"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dr import GRADES, REFERABLE_GRADE  # noqa: E402
from dr.data import FundusDataset, eval_transform, predict_proba  # noqa: E402
from dr.gradcam import GradCAM, overlay  # noqa: E402
from dr.metrics import (decode, expected_grade, optimize_thresholds, patient_bootstrap_ci, quadratic_kappa,  # noqa: E402
                        referable_probability, summarize, summarize_hard)
from dr.model import gradcam_target_layer, load_checkpoint  # noqa: E402

SHORT = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Prolif."]


def run_model(model, frame, data_dir, mean, std, tta=True) -> np.ndarray:
    loader = DataLoader(FundusDataset(frame, data_dir, eval_transform(mean, std)), batch_size=32, num_workers=0)
    out = []
    for x, _ in loader:
        out.append(predict_proba(model, x.contiguous(memory_format=torch.channels_last), tta=tta))
    return torch.cat(out).numpy()


def with_cis(y, probs, thresholds, patients, n_boot) -> dict:
    m = summarize(y, probs, thresholds)
    pred = decode(expected_grade(probs), thresholds)
    ref_p = referable_probability(probs)
    ci = {
        "quadratic_kappa": lambda i: quadratic_kappa(y[i], pred[i]),
        "accuracy": lambda i: float((y[i] == pred[i]).mean()),
        "referable_auc": lambda i: roc_auc_score(y[i] >= REFERABLE_GRADE, ref_p[i]),
        "referable_sensitivity": lambda i: float((pred[i][y[i] >= REFERABLE_GRADE] >= REFERABLE_GRADE).mean()),
        "referable_specificity": lambda i: float((pred[i][y[i] < REFERABLE_GRADE] < REFERABLE_GRADE).mean()),
    }
    m["ci95"] = {k: patient_bootstrap_ci(f, patients, n_boot=n_boot) for k, f in ci.items()}
    return m


def plot_confusion(cm, path, title):
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(5), SHORT, rotation=30, ha="right")
    ax.set_yticks(range(5), SHORT)
    ax.set_xlabel("predicted grade")
    ax.set_ylabel("true grade")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true grade")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y, probs, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for name, truth, score in [("Referable DR (grade >= 2)", y >= REFERABLE_GRADE, referable_probability(probs)),
                               ("Any DR (grade >= 1)", y >= 1, 1 - probs[:, 0])]:
        fpr, tpr, _ = roc_curve(truth, score)
        ax.plot(fpr, tpr, lw=2, label=f"{name}: AUC {roc_auc_score(truth, score):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title("ROC on the held-out test set")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(y, scores, thresholds, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [scores[y == g] for g in range(5)]
    ax.boxplot(data, orientation="horizontal", widths=0.6)
    for t in thresholds:
        ax.axvline(t, color="tab:red", ls="--", lw=1)
    ax.set_yticks(range(1, 6), SHORT)
    ax.set_xlabel("severity score (probability-weighted mean grade); dashed = decision cut-points")
    ax.set_title("Severity score by true grade (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training(log_csv, path):
    log = pd.read_csv(log_csv)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(log.epoch, log.train_loss, label="train loss")
    a1.plot(log.epoch, log.val_loss, label="validation loss")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(log.epoch, log.val_kappa, label="validation quadratic kappa")
    a2.plot(log.epoch, log.val_accuracy_argmax, label="validation accuracy (argmax)")
    best = log.loc[log.val_kappa.idxmax()]
    a2.axvline(best.epoch, color="grey", ls="--", lw=1, label=f"kept epoch {int(best.epoch)}")
    a2.set_xlabel("epoch")
    a2.legend()
    for a in (a1, a2):
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gradcam(model, frame, pred, data_dir, mean, std, path, seed=0):
    """One correctly graded test image per grade: preprocessed input above, Grad-CAM (severity) below."""
    rng = np.random.default_rng(seed)
    cam = GradCAM(model, gradcam_target_layer(model))
    tf = eval_transform(mean, std)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
    for g in range(5):
        idx = np.flatnonzero((frame.grade.to_numpy() == g) & (pred == g))
        if len(idx) == 0:
            idx = np.flatnonzero(frame.grade.to_numpy() == g)
        i = int(rng.choice(idx))
        with Image.open(Path(data_dir) / f"{frame.image.iloc[i]}.jpg") as im:
            image = im.convert("RGB")
        heat = cam(tf(image).unsqueeze(0))
        axes[0, g].imshow(image)
        axes[0, g].set_title(f"true {GRADES[g]}\npredicted {GRADES[int(pred[i])]}", fontsize=10)
        axes[1, g].imshow(overlay(image, heat))
        for a in axes[:, g]:
            a.axis("off")
    cam.remove()
    fig.suptitle("Grad-CAM of the severity score on held-out test images (preprocessed input above, heat map below)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def score_file(name, csv_path, test_df, n_boot):
    """Metrics for another model's predictions (CSV with image + p0..p4, or image + pred_grade) on the test split."""
    other = pd.read_csv(csv_path)
    other = other.drop(columns=[c for c in ("patient", "grade") if c in other.columns])
    merged = test_df[["image", "patient", "grade"]].merge(other, on="image", how="inner")
    y, patients = merged.grade.to_numpy(), merged.patient.to_numpy()
    if all(f"p{g}" in merged for g in range(5)):
        probs = merged[[f"p{g}" for g in range(5)]].to_numpy()
        thresholds = json.loads(merged.thresholds.iloc[0]) if "thresholds" in merged else (0.5, 1.5, 2.5, 3.5)
        m = with_cis(y, probs, thresholds, patients, n_boot)
    else:
        pred = merged.pred_grade.to_numpy()
        m = summarize_hard(y, pred)
        m["ci95"] = {
            "quadratic_kappa": patient_bootstrap_ci(lambda i: quadratic_kappa(y[i], pred[i]), patients, n_boot),
            "accuracy": patient_bootstrap_ci(lambda i: float((y[i] == pred[i]).mean()), patients, n_boot),
        }
    m["name"] = name
    return m


def other_eye_scores(frame: pd.DataFrame, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Severity score of the same patient's other eye (own score where the other eye is not in this split)."""
    lookup = dict(zip(zip(frame.patient, frame.eye), scores))
    swap = {"left": "right", "right": "left"}
    other = np.array([lookup.get((p, swap[e]), np.nan) for p, e in zip(frame.patient, frame.eye)])
    has_other = ~np.isnan(other)
    return np.where(has_other, other, scores), has_other


def both_eyes(val_df, val_scores, test_df, test_scores, n_boot) -> dict:
    """Blend each eye's severity score with the other eye's: alpha * own + (1 - alpha) * other.

    Both eyes of a diabetic patient usually have similar severity, so the other eye is extra evidence (the 2015
    Kaggle winners used it the same way). alpha and the four cut-points are tuned on validation only.
    """
    y_val, y = val_df.grade.to_numpy(), test_df.grade.to_numpy()
    other_val, _ = other_eye_scores(val_df, val_scores)
    other_test, has_other = other_eye_scores(test_df, test_scores)
    best = None
    for alpha in np.round(np.linspace(0.5, 1.0, 11), 2):
        blended = alpha * val_scores + (1 - alpha) * other_val
        cuts = optimize_thresholds(blended, y_val)
        k = quadratic_kappa(y_val, decode(blended, cuts))
        if best is None or k > best[0] + 1e-9:
            best = (k, float(alpha), cuts)
    val_kappa, alpha, cuts = best
    pred = decode(alpha * test_scores + (1 - alpha) * other_test, cuts)
    m = summarize_hard(y, pred)
    patients = test_df.patient.to_numpy()
    ref = lambda i, want: float(((pred[i] >= REFERABLE_GRADE) == want)[(y[i] >= REFERABLE_GRADE) == want].mean())  # noqa: E731
    m["ci95"] = {
        "quadratic_kappa": patient_bootstrap_ci(lambda i: quadratic_kappa(y[i], pred[i]), patients, n_boot),
        "accuracy": patient_bootstrap_ci(lambda i: float((y[i] == pred[i]).mean()), patients, n_boot),
        "referable_sensitivity": patient_bootstrap_ci(lambda i: ref(i, True), patients, n_boot),
        "referable_specificity": patient_bootstrap_ci(lambda i: ref(i, False), patients, n_boot),
    }
    m.update({"alpha": alpha, "thresholds": list(cuts), "validation_kappa": val_kappa,
              "images_with_other_eye": int(has_other.sum())})
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("models/dr_effnet_b0.pt"))
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--compare", action="append", default=[], help='"Name=path/to/predictions.csv"')
    args = p.parse_args()

    torch.set_num_threads(8)
    model, meta = load_checkpoint(args.checkpoint)
    model = model.to(memory_format=torch.channels_last)
    data_dir = args.data or Path("data/processed") / str(meta["image_size"])
    splits = pd.read_csv(args.splits)
    test_df = splits[splits.split == "test"].reset_index(drop=True)
    thresholds = tuple(meta["thresholds"])
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    probs = run_model(model, test_df, data_dir, meta["mean"], meta["std"], tta=not args.no_tta)
    y, patients = test_df.grade.to_numpy(), test_df.patient.to_numpy()
    scores = expected_grade(probs)
    pred = decode(scores, thresholds)
    test_metrics = with_cis(y, probs, thresholds, patients, args.bootstrap)
    test_metrics["n_patients"] = int(test_df.patient.nunique())
    test_metrics["argmax_quadratic_kappa"] = quadratic_kappa(y, probs.argmax(1))
    test_metrics["argmax_accuracy"] = float((probs.argmax(1) == y).mean())

    preds = test_df[["image", "patient", "grade"]].copy()
    for g in range(5):
        preds[f"p{g}"] = probs[:, g].round(5)
    preds["severity_score"] = scores.round(4)
    preds["pred_grade"] = pred
    preds["thresholds"] = json.dumps(list(thresholds))
    preds.to_csv(args.out / "test_predictions.csv", index=False)

    val_df = splits[splits.split == "val"].reset_index(drop=True)
    val_probs = run_model(model, val_df, data_dir, meta["mean"], meta["std"], tta=not args.no_tta)
    eyes = both_eyes(val_df, expected_grade(val_probs), test_df, scores, args.bootstrap)

    comparison = [score_file(*c.split("=", 1), test_df, args.bootstrap) for c in args.compare]
    report = {
        "checkpoint": str(args.checkpoint).replace("\\", "/"),
        "arch": meta["arch"],
        "image_size": meta["image_size"],
        "thresholds": list(thresholds),
        "tta": not args.no_tta,
        "validation": meta.get("val_metrics"),
        "test": test_metrics,
        "test_both_eyes": eyes,
        "comparison": comparison,
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_confusion(test_metrics["confusion_matrix"], figures / "confusion_matrix.png",
                   f"Test set ({len(y)} images, {test_df.patient.nunique()} patients)")
    plot_roc(y, probs, figures / "roc_curves.png")
    plot_scores(y, scores, thresholds, figures / "severity_scores.png")
    log_csv = Path("reports") / f"{args.checkpoint.stem}_log.csv"
    if log_csv.exists():
        plot_training(log_csv, figures / "training_curves.png")
    plot_gradcam(model, test_df, pred, data_dir, meta["mean"], meta["std"], figures / "gradcam_examples.png")

    t = test_metrics
    print(f"TEST ({t['n_images']} images, {t['n_patients']} patients, TTA={not args.no_tta})")
    for k in ["quadratic_kappa", "accuracy", "referable_auc", "referable_sensitivity", "referable_specificity"]:
        lo, hi = t["ci95"][k]
        print(f"  {k:24s} {t[k]:.4f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"  {'any_dr_auc':24s} {t['any_dr_auc']:.4f}")
    print(f"  {'macro_f1':24s} {t['macro_f1']:.4f}")
    print(f"  {'within_one_grade':24s} {t['within_one_grade']:.4f}")
    print(f"  per-grade recall          {np.round(t['per_grade_recall'], 3).tolist()}")
    print(f"  argmax decoding: kappa {t['argmax_quadratic_kappa']:.4f}, accuracy {t['argmax_accuracy']:.4f}")
    print("  confusion:\n  " + str(np.array(t["confusion_matrix"])).replace("\n", "\n  "))
    e = eyes
    print(f"BOTH EYES (alpha {e['alpha']}, cut-points {e['thresholds']}, {e['images_with_other_eye']} test images have "
          f"the other eye): kappa {e['quadratic_kappa']:.4f} (95% CI {e['ci95']['quadratic_kappa'][0]:.3f}-"
          f"{e['ci95']['quadratic_kappa'][1]:.3f}), accuracy {e['accuracy']:.4f}, referable sensitivity "
          f"{e['referable_sensitivity']:.4f}, specificity {e['referable_specificity']:.4f}")
    for c in comparison:
        print(f"{c['name']}: kappa {c['quadratic_kappa']:.4f} accuracy {c['accuracy']:.4f} (n={c['n_images']})")


if __name__ == "__main__":
    main()
