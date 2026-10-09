#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def collect(root: Path):
    # Accept common layouts: Original/Ground Truth or images/masks.
    candidates = [
        (root / "Original", root / "Ground Truth"),
        (root / "CVC-ClinicDB" / "Original", root / "CVC-ClinicDB" / "Ground Truth"),
        (root / "images", root / "masks"),
    ]
    for img_dir, mask_dir in candidates:
        if img_dir.exists() and mask_dir.exists():
            return img_dir, mask_dir
    raise FileNotFoundError(
        f"Could not find CVC-ClinicDB image/mask folders below {root}. "
        "Expected Original + 'Ground Truth' (or images + masks)."
    )


def image_files(path: Path):
    out = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
        out.extend(path.glob(ext))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description="Prepare CVC-ClinicDB as an external test set")
    p.add_argument("--raw-root", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--mask-threshold", type=int, default=127)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    raw = Path(args.raw_root).expanduser().resolve()
    out = Path(args.prepared_root).expanduser().resolve()
    img_dir, mask_dir = collect(raw)

    images = image_files(img_dir)
    masks = image_files(mask_dir)
    im = {x.stem: x for x in images}
    mm = {x.stem: x for x in masks}
    common = sorted(im.keys() & mm.keys())
    missing_masks = sorted(im.keys() - mm.keys())
    missing_images = sorted(mm.keys() - im.keys())
    if not common:
        raise RuntimeError("No image/mask pairs found")
    if missing_masks or missing_images:
        raise RuntimeError(
            f"Unpaired files: images_without_masks={len(missing_masks)}, "
            f"masks_without_images={len(missing_images)}"
        )

    report = {
        "pairs": len(common),
        "source_images": str(img_dir),
        "source_masks": str(mask_dir),
        "target": "polyp foreground",
        "mask_threshold": args.mask_threshold,
        "split_unit": "image; external test only",
        "prepared_root": str(out),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    out_img = out / "images"
    out_mask = out / "masks"
    meta = out / "metadata"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    rows = []
    bad_shapes = []
    empty_masks = []
    for stem in common:
        image = Image.open(im[stem]).convert("RGB")
        mask = Image.open(mm[stem]).convert("L")
        if image.size != mask.size:
            bad_shapes.append((stem, image.size, mask.size))
            continue
        mask_np = np.asarray(mask)
        binary = (mask_np > args.mask_threshold).astype(np.uint8) * 255
        if binary.max() == 0:
            empty_masks.append(stem)
        ip = out_img / f"{stem}.png"
        mp = out_mask / f"{stem}.png"
        image.save(ip)
        Image.fromarray(binary).save(mp)
        rows.append({"sample_id": stem, "image": str(ip), "mask": str(mp)})

    if bad_shapes:
        raise RuntimeError(f"Found {len(bad_shapes)} image/mask shape mismatches; first={bad_shapes[:3]}")
    if empty_masks:
        print(f"WARNING: {len(empty_masks)} empty masks; first={empty_masks[:5]}")

    df = pd.DataFrame(rows)
    df.to_csv(meta / "samples.csv", index=False)
    # External generalization protocol: all cases are test cases.
    df.to_csv(meta / "test_split.csv", index=False)
    report["prepared_pairs"] = len(df)
    report["empty_masks"] = len(empty_masks)
    (meta / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
