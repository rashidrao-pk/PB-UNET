"""ISIC 2016 Task 1: preserve official test images and split training only."""
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from .data import Sample

FOLDERS = {
    "train_images": "ISBI2016_ISIC_Part1_Training_Data",
    "train_masks": "ISBI2016_ISIC_Part1_Training_GroundTruth",
    "test_images": "ISBI2016_ISIC_Part1_Test_Data",
    "test_masks": "ISBI2016_ISIC_Part1_Test_GroundTruth",
}


def _inventory(folder, masks=False):
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing ISIC 2016 folder: {folder}")
    pattern = r"(ISIC_\d+)_segmentation\.png" if masks else r"(ISIC_\d+)\.(?:jpg|jpeg|png)"
    files = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        match = re.fullmatch(pattern, path.name, flags=re.IGNORECASE)
        if not match:
            continue  # Ignore archive readmes and metadata, not image identities.
        key = match[1].upper()
        if key in files:
            raise ValueError(f"Duplicate ISIC image identity in {folder}: {key}")
        files[key] = path
    return files


def source_pairs(config):
    dataset = config["dataset"]
    root = Path(dataset["raw_root"])
    parts = {}
    identities = {}
    for name, default_count in (("train", 900), ("test", 379)):
        images = _inventory(root / FOLDERS[f"{name}_images"])
        masks = _inventory(root / FOLDERS[f"{name}_masks"], masks=True)
        if not images or images.keys() != masks.keys():
            raise ValueError(f"{name}: ISIC image and segmentation-mask identities differ or are empty")
        expected = dataset.get(f"expected_{name}_pairs", default_count)
        if expected is not None and len(images) != expected:
            raise ValueError(f"expected {expected} official {name} pairs, found {len(images)}")
        identities[name] = set(images)
        parts[name] = [Sample(str(images[key]), str(masks[key])) for key in sorted(images)]
    if identities["train"] & identities["test"]:
        raise ValueError("Official ISIC training/test image identities overlap")
    return parts


def read_pair(sample):
    image = cv2.imread(sample.image, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(sample.mask, cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise ValueError(f"Unreadable ISIC image or mask: {sample.image}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 color ISIC image: {sample.image}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask dimensions differ: {sample.image}")
    if not set(np.unique(mask)).issubset({0, 255}) or not mask.any():
        raise ValueError(f"Expected nonempty binary 0/255 lesion mask: {sample.mask}")
    return image, (mask > 127).astype(np.uint8) * 255


def validate_source(config):
    parts = source_pairs(config)
    hashes = {}
    for name, samples in parts.items():
        hashes[name] = set()
        for sample in tqdm(samples, desc=f"Validate ISIC {name}", unit="image"):
            read_pair(sample)
            hashes[name].add(hashlib.sha256(Path(sample.image).read_bytes()).hexdigest())
    if hashes["train"] & hashes["test"]:
        raise ValueError("Official ISIC training/test contain byte-identical images")
    return parts


def prepare_isic2016(config, dry_run=False):
    source = validate_source(config)
    split = config.get("split", {})
    seed = split.get("seed", 42)
    fraction = split.get("validation", .2)
    if not 0 < fraction < 1:
        raise ValueError("validation must be a fraction of the official training set between 0 and 1")
    n_val = int(len(source["train"]) * fraction)
    if not 0 < n_val < len(source["train"]):
        raise ValueError("Validation fraction must leave nonempty training and validation sets")
    train, val = train_test_split(source["train"], test_size=n_val, random_state=seed)
    parts = {"train": train, "val": val, "test": source["test"]}
    report = dict(pairs=sum(map(len, source.values())), train=len(train), validation=len(val),
                  test=len(parts["test"]), official_test_preserved=True, seed=seed,
                  validation_fraction_of_official_train=fraction,
                  split_unit="image; patient/lesion linkage metadata unavailable",
                  target="skin lesion foreground", mask_threshold=127,
                  preparation="decoded RGB intensities retained; lossless PNG; no resize or normalization")
    if dry_run:
        return {**report, "source_valid": True}
    raw = Path(config["dataset"]["raw_root"]).resolve()
    out = Path(config["dataset"]["prepared_root"]).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("prepared_root must be outside raw_root")
    for folder in ("images", "masks", "metadata"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    assignments = {sample.image: name for name, samples in parts.items() for sample in samples}
    rows = []
    for source_name, samples in source.items():
        for sample in tqdm(samples, desc=f"Prepare ISIC {source_name}", unit="image"):
            image, mask = read_pair(sample)
            stem = Path(sample.image).stem.upper()
            image_path, mask_path = (out / folder / f"{stem}.png" for folder in ("images", "masks"))
            for path, array in ((image_path, image), (mask_path, mask)):
                if not cv2.imwrite(str(path), array):
                    raise OSError(f"Failed to save {path}")
            rows.append(dict(image=str(image_path), mask=str(mask_path), image_id=stem,
                             source_image=str(Path(sample.image).resolve()), source_mask=str(Path(sample.mask).resolve()),
                             source_image_sha256=hashlib.sha256(Path(sample.image).read_bytes()).hexdigest(),
                             source_mask_sha256=hashlib.sha256(Path(sample.mask).read_bytes()).hexdigest(),
                             official_split=source_name, split=assignments[sample.image],
                             height=image.shape[0], width=image.shape[1], foreground_pixels=int((mask > 0).sum())))
    with (out / "metadata/manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, samples in parts.items():
        with (out / f"metadata/{name}_split.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "mask", "group"])
            for sample in samples:
                stem = Path(sample.image).stem.upper()
                writer.writerow([str(out / "images" / f"{stem}.png"), str(out / "masks" / f"{stem}.png"), ""])
    report["prepared_root"] = str(out)
    (out / "metadata/preparation.json").write_text(json.dumps(report, indent=2))
    return report
