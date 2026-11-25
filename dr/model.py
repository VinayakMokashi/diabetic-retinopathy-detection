"""Model construction and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import timm
import torch

from . import NUM_CLASSES


def create_model(arch: str, pretrained: bool = True, drop_rate: float = 0.3) -> torch.nn.Module:
    """An ImageNet-pretrained timm backbone with a new 5-way classification head."""
    return timm.create_model(arch, pretrained=pretrained, num_classes=NUM_CLASSES, drop_rate=drop_rate)


def save_checkpoint(path: str | Path, model: torch.nn.Module, meta: dict, half: bool = True) -> None:
    """Save weights plus everything inference needs (architecture, image size, normalisation, cut-points).

    Weights are stored in float16 by default to halve the file size; they are cast back to float32 on load.
    """
    state = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, **meta}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = create_model(checkpoint["arch"], pretrained=False, drop_rate=0.0)
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state)
    model.eval().to(device)
    meta = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model, meta


def gradcam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """The last convolutional feature block, whose activations Grad-CAM weights."""
    for name in ("bn2", "conv_head", "head_hidden", "final_conv"):   # EfficientNet / MobileNet / RegNet families
        if hasattr(model, name) and isinstance(getattr(model, name), torch.nn.Module):
            return getattr(model, name)
    for name in ("layer4", "stages", "features"):                     # ResNet / ConvNeXt / DenseNet families
        if hasattr(model, name):
            module = getattr(model, name)
            return module[-1] if isinstance(module, torch.nn.Sequential) else module
    raise ValueError(f"no Grad-CAM layer known for {type(model).__name__}")
