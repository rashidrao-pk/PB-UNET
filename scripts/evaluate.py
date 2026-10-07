#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portable_bridge_unet.data import Sample, make_loader
from portable_bridge_unet.evaluation import evaluate_loader
from portable_bridge_unet.checkpoints import load_checkpoint
from portable_bridge_unet.plotting import plot_evaluation_metrics
from portable_bridge_unet.config import parse_config_args
from portable_bridge_unet.utils import resolve_device, save_json


def read_split(path: str) -> list[Sample]:
    df = pd.read_csv(path)
    return [
        Sample(str(r.image), str(r.mask), None if pd.isna(getattr(r, "group", None)) or str(getattr(r, "group", "")) == "" else str(r.group))
        for r in df.itertuples(index=False)
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Checkpoint path (default: config)")
    p.add_argument("--split-csv", default=None, help="Split CSV path (default: config)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--postprocess", action="store_true")
    p.add_argument("--no-hd95", action="store_true")
    p.add_argument("--out-dir", default="evaluation")
    args = parse_config_args(p, "evaluate")

    device = resolve_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    threshold = ckpt.get("threshold", 0.5) if args.threshold is None else args.threshold

    samples = read_split(args.split_csv)
    loader = make_loader(
        samples, args.batch_size,
        image_size=int(ckpt.get("image_size", 256)),
        channels=int(ckpt.get("in_channels", 3)),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    df, summary = evaluate_loader(
        model, loader, device, threshold,
        apply_postprocessing=args.postprocess,
        compute_hd95=not args.no_hd95,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "postprocessed" if args.postprocess else "raw"
    df.to_csv(out / f"per_image_{suffix}.csv", index=False)
    save_json(summary, out / f"summary_{suffix}.json")
    plot_evaluation_metrics(df, out, suffix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
