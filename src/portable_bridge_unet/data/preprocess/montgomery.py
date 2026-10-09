"""Montgomery lung preparation with both lungs retained and fixed shared splits."""
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def source_pairs(config):
    root = Path(config["dataset"]["raw_root"])
    folders = [root / name for name in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask")]
    inventories = []
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing Montgomery folder: {folder}")
        inventories.append({p.name: p for p in sorted(folder.glob("*.png"))})
    names = set(inventories[0])
    if not names or any(set(files) != names for files in inventories[1:]):
        raise ValueError("Montgomery images and left/right masks must have identical nonempty filename sets")
    expected = config["dataset"].get("expected_pairs", 138)
    if expected is not None and len(names) != expected:
        raise ValueError(f"expected {expected} Montgomery images, found {len(names)}")
    pairs, patients = [], set()
    for name in sorted(names):
        match = re.fullmatch(r"MCUCXR_(\d+)_([01])\.png", name)
        if not match:
            raise ValueError(f"Unexpected Montgomery filename: {name}")
        patient = str(int(match[1]))
        if patient in patients:
            raise ValueError(f"Duplicate Montgomery patient ID: {patient}")
        patients.add(patient)
        pairs.append((*(files[name] for files in inventories), patient, int(match[2])))
    return pairs


def read_pair(pair):
    arrays = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in pair[:3]]
    if any(array is None for array in arrays):
        raise ValueError(f"Unreadable Montgomery image/mask: {pair[0]}")
    image, left, right = arrays
    if image.dtype not in (np.uint8, np.uint16) or image.ndim != 2:
        raise ValueError(f"Expected uint8/uint16 grayscale chest X-ray: {pair[0]}")
    for mask in (left, right):
        if mask.shape != image.shape:
            raise ValueError(f"Image/mask dimensions differ: {pair[0]}")
        if not set(np.unique(mask)).issubset({0, 1, 255}) or not mask.any():
            raise ValueError(f"Expected nonempty binary lung masks: {pair[0]}")
    # The NLM readme describes 12-bit acquisition; preserve uint8 exports and
    # explicitly map higher-bit-depth PNGs to the loader's uint8 convention.
    if image.dtype == np.uint16:
        lower, upper = int(image.min()), int(image.max())
        image = (np.round((image.astype(np.float32) - lower) * (255.0 / (upper - lower))).astype(np.uint8)
                 if upper > lower else np.zeros_like(image, dtype=np.uint8))
    return image, ((left > 0) | (right > 0)).astype(np.uint8) * 255


def prepare_montgomery(config, dry_run=False):
    pairs = source_pairs(config)
    # Check every source and split feasibility before creating output files.
    for pair in tqdm(pairs, desc="Validate Montgomery", unit="image"):
        read_pair(pair)
    split = config.get("split", {})
    seed = split.get("seed", 42)
    n_val = int(len(pairs) * split.get("validation", .15))
    n_test = int(len(pairs) * split.get("test", .15))
    if min(n_val, n_test) < 1 or n_val + n_test >= len(pairs):
        raise ValueError("Split fractions must leave nonempty train, validation and test sets")
    remaining, val = train_test_split(pairs, test_size=n_val, random_state=seed,
                                      stratify=[p[4] for p in pairs])
    train, test = train_test_split(remaining, test_size=n_test, random_state=seed,
                                  stratify=[p[4] for p in remaining])
    report = dict(pairs=len(pairs), train=len(train), validation=len(val), test=len(test),
                  seed=seed, split_unit="unique filename ID; one image per ID; independent patient metadata unavailable",
                  stratification="normal/abnormal filename suffix", target="union of left and right lungs",
                  mask_threshold=0, preparation="uint8 grayscale preserved; uint16 per-image min-max to uint8; PNG; no resize",
                  postprocessing="disabled; retain both lungs")
    if dry_run:
        return {**report, "source_valid": True}
    raw = Path(config["dataset"]["raw_root"]).resolve()
    out = Path(config["dataset"]["prepared_root"]).resolve()
    if out == raw or raw in out.parents:
        raise ValueError("prepared_root must be outside raw_root")
    for folder in ("images", "masks", "metadata"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in tqdm(pairs, desc="Prepare Montgomery", unit="image"):
        image, mask = read_pair(pair)
        image_path, mask_path = (out / folder / pair[0].name for folder in ("images", "masks"))
        for path, array in ((image_path, image), (mask_path, mask)):
            if not cv2.imwrite(str(path), array):
                raise OSError(f"Failed to save {path}")
        row = dict(image=str(image_path), mask=str(mask_path), group=pair[3], abnormal=pair[4],
                   height=image.shape[0], width=image.shape[1], foreground_pixels=int((mask > 0).sum()))
        source_image = cv2.imread(str(pair[0]), cv2.IMREAD_UNCHANGED)
        row.update(source_dtype=str(source_image.dtype), source_min=int(source_image.min()),
                   source_max=int(source_image.max()),
                   intensity_mapping="preserved" if source_image.dtype == np.uint8 else "per-image min-max uint8")
        for label, path in zip(("image", "left_mask", "right_mask"), pair[:3]):
            row[f"source_{label}"] = str(path.resolve())
            row[f"source_{label}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    with (out / "metadata/manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, part in zip(("train", "val", "test"), (train, val, test)):
        with (out / f"metadata/{name}_split.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "mask", "group", "abnormal"])
            writer.writerows((str(out / "images" / p[0].name), str(out / "masks" / p[0].name), p[3], p[4]) for p in part)
    report["prepared_root"] = str(out)
    (out / "metadata/preparation.json").write_text(json.dumps(report, indent=2))
    return report
