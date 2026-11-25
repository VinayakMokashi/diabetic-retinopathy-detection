"""Single-image inference used by the CLI and the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import GRADES, REFERABLE_GRADE
from .data import eval_transform, predict_proba
from .gradcam import GradCAM, overlay
from .metrics import decode, expected_grade
from .model import gradcam_target_layer, load_checkpoint
from .preprocessing import preprocess_fundus


@dataclass
class Prediction:
    grade: int
    grade_name: str
    probabilities: list[float]
    severity_score: float
    referable_probability: float
    referable: bool
    preprocessed: Image.Image

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "preprocessed"}


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.model, self.meta = load_checkpoint(checkpoint, device)
        self.device = device
        self.size = int(self.meta["image_size"])
        self.thresholds = tuple(self.meta["thresholds"])
        self.transform = eval_transform(self.meta["mean"], self.meta["std"])

    def preprocess(self, image: Image.Image) -> Image.Image:
        return preprocess_fundus(image, size=self.size)

    def predict(self, image: Image.Image, tta: bool = True) -> Prediction:
        pre = self.preprocess(image)
        x = self.transform(pre).unsqueeze(0).to(self.device)
        probs = predict_proba(self.model, x, tta=tta).cpu().numpy()
        score = float(expected_grade(probs)[0])
        grade = int(decode(np.array([score]), self.thresholds)[0])
        referable_p = float(probs[0, REFERABLE_GRADE:].sum())
        return Prediction(grade=grade, grade_name=GRADES[grade], probabilities=[float(p) for p in probs[0]],
                          severity_score=score, referable_probability=referable_p,
                          referable=grade >= REFERABLE_GRADE, preprocessed=pre)

    def explain(self, image: Image.Image, target: int | str = "severity") -> tuple[Image.Image, np.ndarray]:
        """Grad-CAM heat map over the preprocessed image."""
        pre = self.preprocess(image)
        x = self.transform(pre).unsqueeze(0).to(self.device)
        cam = GradCAM(self.model, gradcam_target_layer(self.model))
        try:
            heat = cam(x, target=target)
        finally:
            cam.remove()
        return overlay(pre, heat), heat
