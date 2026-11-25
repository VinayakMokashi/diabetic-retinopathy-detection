"""Fine-tune an ImageNet-pretrained CNN to grade diabetic retinopathy (0-4).

Training uses cross-entropy over the five grades. Predictions are decoded as the probability-weighted mean grade cut
at four thresholds, which respects the ordering of the grades; the thresholds are tuned on the validation set after
training. The epoch with the best validation quadratic kappa is kept. The test set is never touched here.

Usage:
    python scripts/train.py --arch efficientnet_b0.ra_in1k --size 384 --epochs 20 --out models/dr_effnet_b0.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dr import GRADES  # noqa: E402
from dr.data import FundusDataset, channel_stats, eval_transform, predict_proba, train_transform  # noqa: E402
from dr.metrics import decode, expected_grade, optimize_thresholds, quadratic_kappa, summarize  # noqa: E402
from dr.model import create_model, save_checkpoint  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect(model, loader, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(predict_proba(model, x.contiguous(memory_format=torch.channels_last), tta=tta))
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="efficientnet_b0.ra_in1k")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--data", type=Path, default=None, help="cache from preprocess_images.py (default data/processed/<size>)")
    p.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--drop-rate", type=float, default=0.3)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("models/dr_model.pt"))
    p.add_argument("--log", type=Path, default=None, help="per-epoch CSV log (default reports/<out stem>_log.csv)")
    p.add_argument("--limit", type=int, default=None, help="use only this many train/val images (smoke test)")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="recompute activations in the backward pass: ~2x less RAM, ~20%% slower")
    args = p.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    data_dir = args.data or Path("data/processed") / str(args.size)
    splits = pd.read_csv(args.splits)
    train_df, val_df = splits[splits.split == "train"], splits[splits.split == "val"]
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=args.seed)
        val_df = val_df.sample(n=min(args.limit, len(val_df)), random_state=args.seed)
    print(f"train {len(train_df)} images / val {len(val_df)} images | arch {args.arch} @ {args.size}px", flush=True)

    mean, std = channel_stats([data_dir / f"{n}.jpg" for n in train_df.image])
    print("normalisation (training-set channel stats):", mean, std, flush=True)
    train_loader = DataLoader(FundusDataset(train_df, data_dir, train_transform(mean, std)),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    # Validation decodes 411 small JPEGs per epoch; loading them in the main process saves two worker processes
    # (about 0.4 GB of RAM each on Windows) at a negligible cost in speed.
    val_loader = DataLoader(FundusDataset(val_df, data_dir, eval_transform(mean, std)), batch_size=32, num_workers=0)

    model = create_model(args.arch, pretrained=True, drop_rate=args.drop_rate).to(memory_format=torch.channels_last)
    if args.grad_checkpointing:
        model.set_grad_checkpointing(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_epochs * len(train_loader))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    log, best_kappa, best_state, best_epoch = [], -1.0, None, -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x = x.contiguous(memory_format=torch.channels_last)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)
        train_time = time.time() - start

        probs, y_val = collect(model, val_loader, tta=False)
        val_loss = float(torch.nn.functional.nll_loss(torch.log(torch.from_numpy(probs) + 1e-8),
                                                      torch.from_numpy(y_val)))
        kappa = quadratic_kappa(y_val, decode(expected_grade(probs)))
        acc = float((probs.argmax(1) == y_val).mean())
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": running / seen,
               "val_loss": val_loss, "val_kappa": kappa, "val_accuracy_argmax": acc, "epoch_seconds": train_time}
        log.append(row)
        flag = ""
        if kappa > best_kappa:
            best_kappa, best_epoch, best_state = kappa, epoch, copy.deepcopy(model.state_dict())
            flag = "  <- best"
        print(f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val kappa {kappa:.4f} | val acc {acc:.4f} | {train_time:.0f}s{flag}", flush=True)

    log_path = args.log or Path("reports") / f"{args.out.stem}_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(log_path, index=False)

    model.load_state_dict(best_state)
    probs, y_val = collect(model, val_loader, tta=True)
    scores = expected_grade(probs)
    thresholds = optimize_thresholds(scores, y_val)
    val_metrics = summarize(y_val, probs, thresholds)
    print(f"best epoch {best_epoch}; validation with TTA: kappa {val_metrics['quadratic_kappa']:.4f} "
          f"(default cut-points {quadratic_kappa(y_val, decode(scores)):.4f}), thresholds {thresholds}", flush=True)

    meta = {
        "arch": args.arch,
        "image_size": args.size,
        "mean": mean,
        "std": std,
        "thresholds": list(thresholds),
        "grades": GRADES,
        "best_epoch": best_epoch,
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
        "train_args": {k: str(v) for k, v in vars(args).items()},
        "timm_version": str(timm.__version__),
        "torch_version": str(torch.__version__),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, model, meta)
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
