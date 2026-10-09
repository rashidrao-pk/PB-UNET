#!/usr/bin/env python3
"""Combine saved dataset smoke previews and descriptions into one figure."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATASETS = {
    "cvc_clinicdb": (
        "CVC-ClinicDB — External Colonoscopy",
        "RGB colonoscopy images with manual polyp masks. In this suite the entire "
        "dataset is an external test cohort for Kvasir-trained checkpoints, "
        "with no target training or fine-tuning.",
    ),
    "brats2020": (
        "BraTS 2020 — Brain MRI",
        "Labeled glioma MRI cases. This experiment uses axial FLAIR slices and "
        "binary whole-tumor masks, with cases separated before slicing. "
        "It is a 2D internal holdout experiment, not the official 3D challenge protocol.",
    ),
    "brats2021": (
        "BraTS 2021 — Brain MRI",
        "A separate release of labeled glioma MRI cases. Axial FLAIR slices "
        "are used for whole-tumor segmentation, with case-separated holdouts. "
        "This release must not be assumed independent of BraTS 2020.",
    ),
    "drive": (
        "DRIVE — Retinal Fundus",
        "RGB retinal fundus photographs for blood-vessel segmentation. "
        "First manual annotations define vessels; field-of-view masks define "
        "the evaluation domain. The official test split is preserved.",
    ),
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


def create_dataset_overview(smoke_dir, out_dir, sample_index=0, datasets=None):
    """Write PNG/PDF plus a source manifest from saved smoke previews.

    Returns output paths; requires usable saved samples for all selected datasets.
    Does not require original images, weights, or a command-line invocation.
    """
    datasets = list(DATASETS) if datasets is None else list(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - DATASETS.keys():
        raise ValueError("datasets must be a nonempty list of unique supported names")
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    selected = find_previews(Path(smoke_dir), sample_index, datasets)
    out_dir = Path(out_dir)
    fig = plt.figure(figsize=(15, 3 * len(datasets) + 1), facecolor="white")
    grid = fig.add_gridspec(len(datasets), 2, width_ratios=[1, 2.7], hspace=.20, wspace=.08)
    for row, name in enumerate(datasets):
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
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"dataset_overview.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest = {name: {**source, "title": DATASETS[name][0],
                        "description": DATASETS[name][1]} for name, source in selected.items()}
    (out_dir / "dataset_overview_sources.json").write_text(json.dumps(manifest, indent=2))

    return {extension: str((out_dir / f"dataset_overview.{extension}").resolve()) for extension in ("png", "pdf")}
