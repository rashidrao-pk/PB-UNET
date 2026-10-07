#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portable_bridge_unet.data import Sample, make_loader, pair_by_stem, split_samples_legacy
from portable_bridge_unet.metrics import batch_metrics_from_logits
from portable_bridge_unet.models import build_model, count_trainable_parameters
from portable_bridge_unet.config import parse_config_args
from portable_bridge_unet.utils import resolve_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline or Portable-Bridge U-Net in PyTorch")
    p.add_argument("--images", default=None, help="Input image glob (default: dataset.images in config)")
    p.add_argument("--masks", default=None, help="Binary mask glob (default: dataset.masks in config)")
    p.add_argument("--model", choices=["unet", "portable_bridge"], default="portable_bridge")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--channels", type=int, choices=[1, 3], default=3)
    p.add_argument("--filters", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto, cuda, mps, cpu, cuda:0, ...")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr-factor", type=float, default=0.05)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-dir", default="runs/paper_legacy")
    return parse_config_args(p, "train")


def write_split(samples: list[Sample], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mask", "group"])
        for s in samples:
            w.writerow([s.image, s.mask, s.group or ""])


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, threshold=0.5):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    n = 0
    iterator = tqdm(loader, leave=False, desc="train" if training else "val")

    amp_enabled = scaler is not None and scaler.is_enabled()
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        bs = images.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=amp_device, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        metrics = batch_metrics_from_logits(logits.detach(), masks, threshold)
        sums["loss"] += float(loss.detach()) * bs
        for k in metrics:
            sums[k] += float(metrics[k].sum().cpu())
        n += bs
        iterator.set_postfix(loss=f"{sums['loss']/max(n,1):.4f}", dice=f"{sums['dice']/max(n,1):.4f}")

    return {k: v / max(n, 1) for k, v in sums.items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = sorted(glob(args.images, recursive=True))
    masks = sorted(glob(args.masks, recursive=True))
    if not images or not masks:
        raise SystemExit(
            "Prepared image/mask files are missing. Run:\n"
            f"  python scripts/prepare_sunnybrook.py --config {args.config}\n"
            f"  python scripts/check_dataset.py --config {args.config} --require-prepared\n"
            "Then rerun training. Raw DICOM files cannot be loaded as PNG pairs."
        )
    samples = pair_by_stem(images, masks)
    train_s, val_s, test_s = split_samples_legacy(samples, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed)
    write_split(train_s, out / "train_split.csv")
    write_split(val_s, out / "val_split.csv")
    write_split(test_s, out / "test_split.csv")

    print(f"device={device}")
    print(f"samples={len(samples)} train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    train_loader = make_loader(
        train_s, args.batch_size, args.image_size, args.channels,
        shuffle=True, augment=args.augment, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = make_loader(
        val_s, args.batch_size, args.image_size, args.channels,
        shuffle=False, augment=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, in_channels=args.channels, filters=args.filters).to(device)
    print(f"trainable_parameters={count_trainable_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.patience)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    history = []
    best_val = math.inf
    stale = 0
    best_path = out / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, optimizer, scaler, args.threshold)
        val_m = run_epoch(model, val_loader, criterion, device, None, None, args.threshold)
        scheduler.step(val_m["loss"])

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_m['loss']:.5f} val_loss={val_m['loss']:.5f} "
            f"val_dice={val_m['dice']:.5f} val_iou={val_m['iou']:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "in_channels": args.channels,
                    "filters": args.filters,
                    "image_size": args.image_size,
                    "threshold": args.threshold,
                    "epoch": epoch,
                    "val_metrics": val_m,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after {args.patience} epochs without val-loss improvement")
                break

        save_json(history, out / "history.json")

    save_json(history, out / "history.json")
    save_json(vars(args), out / "config.json")
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
