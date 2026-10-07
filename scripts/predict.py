#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portable_bridge_unet.models import build_model
from portable_bridge_unet.postprocess import postprocess
from portable_bridge_unet.utils import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--postprocess", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    channels = int(ckpt.get("in_channels", 3))
    size = int(ckpt.get("image_size", 256))
    threshold = ckpt.get("threshold", 0.5) if args.threshold is None else args.threshold

    model = build_model(ckpt["model_name"], channels, filters=ckpt.get("filters", [32, 64, 128]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    image = cv2.imread(args.image, flag)
    if image is None:
        raise FileNotFoundError(args.image)
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if channels == 1:
        tensor = torch.from_numpy(image[None, None])
    else:
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))[None])

    with torch.no_grad():
        logits = model(tensor.to(device))
        pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= threshold).astype(np.uint8) * 255
    if args.postprocess:
        pred = postprocess(pred)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, pred)
    print(args.output)


if __name__ == "__main__":
    main()
