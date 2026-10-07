#!/usr/bin/env python3
"""Combine saved dataset smoke previews and descriptions into one figure."""
import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "sunnybrook": (
        "Sunnybrook Cardiac MRI",
        "Cardiac cine MRI used to study heart anatomy and function. "
        "This experiment segments the left ventricular (LV) blood cavity "
        "using masks rasterized from manual inner contours.",
    ),
    "busi": (
        "BUSI — Breast Ultrasound",
        "Breast ultrasound images with benign, malignant, and normal categories. "
        "This experiment uses lesion-containing images and segments the annotated "
        "breast lesion; normal images are excluded.",
    ),
    "kvasir_seg": (
        "Kvasir-SEG — Colonoscopy",
        "Color gastrointestinal endoscopy images with manually annotated polyps. "
        "The segmentation target is the polyp foreground, supporting automatic "
        "localization and delineation of colorectal polyps.",
    ),
    "montgomery": (
        "Montgomery — Lung X-rays",
        "Posterior-anterior chest X-rays from tuberculosis screening, including "
        "normal and abnormal cases. This experiment segments both lung fields "
        "using the union of manual left- and right-lung masks. "
        "The target is lung anatomy, not tuberculosis lesions.",
    ),
    "isic2016": (
        "ISIC 2016 — Skin Dermoscopy",
        "RGB dermoscopic photographs of skin lesions with expert binary lesion "
        "outlines. This experiment segments the lesion boundary and foreground, "
        "rather than predicting a diagnosis. The official test set is retained "
        "separately from training and validation.",
    ),
}


def find_previews(root, sample_index, datasets=None):
    """Choose the latest saved report with a usable preview for each dataset."""
    selected = {}
    datasets = list(DATASETS) if datasets is None else datasets
    for path in sorted(root.rglob("report.json")):
        report = json.loads(path.read_text())
        name = report.get("dataset", {}).get("dataset_name")
        samples = report.get("samples", [])
        if name not in datasets or len(samples) <= sample_index:
            continue
        sample = samples[sample_index]
        preview = path.parent / sample["preview"]
        if not preview.is_file():
            continue
        stamp = report.get("created_utc", path.parent.name)
        if name not in selected or stamp > selected[name]["created_utc"]:
            selected[name] = dict(created_utc=stamp, report=str(path.resolve()),
                                  preview=str(preview.resolve()), sample=sample,
                                  prepared_pairs=report["dataset"].get("prepared_pairs"))
    missing = set(datasets) - selected.keys()
    if missing:
        raise ValueError("Missing saved previews for: " + ", ".join(sorted(missing)) +
                         ". Run python scripts/preview_datasets.py first.")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "runs/smoke_test")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Zero-based sample index within each saved report")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/dataset_overview")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                        help="Datasets to include, in row order (default: all five)")
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    try:
        selected = find_previews(args.smoke_dir, args.sample_index, args.datasets)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    fig = plt.figure(figsize=(15, 3 * len(args.datasets) + 1), facecolor="white")
    grid = fig.add_gridspec(len(args.datasets), 2, width_ratios=[1, 2.7], hspace=.20, wspace=.08)
    for row, name in enumerate(args.datasets):
        title, description = DATASETS[name]
        source = selected[name]
        text_ax = fig.add_subplot(grid[row, 0])
        text_ax.axis("off")
        text_ax.text(0, .94, title, fontsize=13, weight="bold", va="top",
                     transform=text_ax.transAxes)
        text_ax.text(0, .76, textwrap.fill(description, 43), fontsize=11, va="top",
                     linespacing=1.5, transform=text_ax.transAxes)
        count = source["prepared_pairs"]
        if count is not None:
            text_ax.text(0, .06, f"Prepared image–mask pairs: {count:,}", fontsize=10,
                         color="#444444", transform=text_ax.transAxes)
        image_ax = fig.add_subplot(grid[row, 1])
        image_ax.imshow(mpimg.imread(source["preview"]))
        image_ax.axis("off")
    fig.suptitle("Datasets and segmentation targets", fontsize=20, weight="bold", y=.97)
    fig.text(.5, .025,
             "Example prepared images • Binary annotation masks • Red annotation overlays\n"
             "Previews use configured resizing and normalization, without augmentation. These are ground-truth masks, not predictions.",
             ha="center", fontsize=10, linespacing=1.5)
    fig.subplots_adjust(top=.91, bottom=.09, left=.035, right=.985)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = args.out_dir / f"dataset_overview.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(path.resolve())
    plt.close(fig)
    manifest = {name: {**source, "title": DATASETS[name][0],
                        "description": DATASETS[name][1]} for name, source in selected.items()}
    (args.out_dir / "dataset_overview_sources.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
