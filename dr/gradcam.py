"""Grad-CAM (Selvaraju et al., 2017) for the grading network.

By default the explained quantity is the *severity score*, the probability-weighted mean grade that the decision
rule thresholds, so the heat map shows which regions pushed the predicted severity up.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output
        output.register_hook(lambda grad: setattr(self, "gradients", grad))

    def __call__(self, x: torch.Tensor, target: int | str = "severity") -> np.ndarray:
        """Heat map in [0, 1] with the spatial size of ``x``. ``target`` is a class index or ``"severity"``."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            if target == "severity":
                probs = torch.softmax(logits, dim=1)
                score = (probs * torch.arange(probs.shape[1], device=probs.device)).sum()
            else:
                score = logits[0, int(target)]
            score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()

    def remove(self) -> None:
        self.handle.remove()


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a colour-mapped heat map over the image."""
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    base = np.asarray(image.convert("RGB").resize(cam.shape[::-1]), dtype=np.float32) / 255.0
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))
