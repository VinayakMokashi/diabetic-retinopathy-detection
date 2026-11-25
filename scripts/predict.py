"""Grade one or more fundus photographs from the command line.

Usage:
    python scripts/predict.py path/to/fundus.jpg [more.jpg ...] [--gradcam out_dir] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dr import GRADES  # noqa: E402
from dr.inference import Predictor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "dr_effnet_b0.pt")
    p.add_argument("--gradcam", type=Path, default=None, help="folder to write Grad-CAM overlays into")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--json", action="store_true", help="print one JSON object per image")
    args = p.parse_args()

    predictor = Predictor(args.checkpoint)
    if args.gradcam:
        args.gradcam.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        with Image.open(path) as im:
            image = im.convert("RGB")
        result = predictor.predict(image, tta=not args.no_tta)
        if args.json:
            print(json.dumps({"image": str(path), **result.as_dict()}))
        else:
            probs = "  ".join(f"{g}:{p:.2f}" for g, p in enumerate(result.probabilities))
            print(f"{path.name}: grade {result.grade} ({GRADES[result.grade]}), severity {result.severity_score:.2f}, "
                  f"P(referable) {result.referable_probability:.2f} | {probs}")
        if args.gradcam:
            overlay, _ = predictor.explain(image)
            overlay.save(args.gradcam / f"{path.stem}_gradcam.png")


if __name__ == "__main__":
    main()
