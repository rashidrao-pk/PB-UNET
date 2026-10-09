"""Prepare Sunnybrook LV cavity masks from inner contours and cine short-axis MRI."""
import csv
import re
from pathlib import Path
import cv2
import numpy as np
import pydicom


def prepare_sunnybrook(config, dry_run=False):
    """Common preparation API; preserve the existing Sunnybrook implementation."""
    if dry_run:
        plan = plan_samples(config)
        return {"pairs": len(plan), "patients": len({p[0] for p in plan})}
    return prepare_dataset(config)

def normalize_id(value):
    return "-".join(str(int(p)) if p.isdigit() else p for p in value.split("-"))

def rasterize_contour(path, shape):
    points = np.loadtxt(path, ndmin=2)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"invalid contour: {path}")
    h, w = shape
    if (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
        raise ValueError(f"contour outside image bounds: {path}")
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask

def plan_samples(config):
    dataset = config["dataset"]
    root = Path(dataset["raw_root"])
    with (root / "scd_patientdata.csv").open(encoding="utf-8-sig", newline="") as stream:
        mapping = {normalize_id(row["OriginalID"]): row["PatientID"] for row in csv.DictReader(stream)}
    contours = sorted((root / "SCD_ManualContours").rglob("*-icontour-manual.txt"))
    if not contours:
        raise ValueError("no inner contours found")
    by_patient = {}
    for contour in contours:
        original = contour.relative_to(root / "SCD_ManualContours").parts[0]
        patient = mapping[normalize_id(original)]
        number = int(re.fullmatch(r"IM-\d+-(\d+)-icontour-manual.txt", contour.name).group(1))
        by_patient.setdefault(patient, []).append((number, contour))
    plan = []
    overrides = dataset.get("series_overrides", {})
    for patient, annotations in sorted(by_patient.items()):
        series_pattern = overrides.get(patient, "CINESAX_*")
        directories = list(root.glob(f"SCD_IMAGES_*/{patient}/{series_pattern}"))
        candidates = []
        for folder in directories:
            frames = {}
            for path in folder.glob("*.dcm"):
                number = int(path.stem.split("-")[-1])
                if number in frames:
                    raise ValueError(f"duplicate frame in {folder}: {number}")
                frames[number] = path
            if all(n in frames for n, _ in annotations):
                if patient not in overrides or folder.name == overrides[patient]:
                    candidates.append(frames)
        if len(candidates) != 1:
            raise ValueError(f"{patient}: expected one matching CINESAX series, found {len(candidates)}; set dataset.series_overrides")
        frames = candidates[0]
        for number, contour in annotations:
            plan.append((patient, number, frames[number], contour))
    return plan

def prepare_dataset(config):
    plan = plan_samples(config)  # Resolve every match before writing any output.
    out = Path(config["dataset"]["prepared_root"])
    for name in ("images", "masks"):
        (out / name).mkdir(parents=True, exist_ok=True)
    rows = []
    for patient, number, dicom, contour in plan:
        ds = pydicom.dcmread(dicom)
        if str(ds.PatientID) != patient or int(ds.InstanceNumber) != number:
            raise ValueError(f"DICOM identity mismatch: {dicom}")
        pixels = ds.pixel_array.astype(np.float32)
        if pixels.ndim != 2 or not np.isfinite(pixels).all():
            raise ValueError(f"invalid pixel array: {dicom}")
        mask = rasterize_contour(contour, pixels.shape)
        pixels = pixels * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
        span = float(np.ptp(pixels))
        image = np.zeros(pixels.shape, dtype=np.uint8) if span == 0 else np.round((pixels - pixels.min()) / span * 255).astype(np.uint8)
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            image = 255 - image
        stem = f"{patient}_{number:04d}"
        image_path, mask_path = out / "images" / f"{stem}.png", out / "masks" / f"{stem}.png"
        for path, array in ((image_path, image), (mask_path, mask)):
            if not cv2.imwrite(str(path), array):
                raise OSError(f"failed to write {path}")
        rows.append(dict(image=str(image_path.resolve()), mask=str(mask_path.resolve()),
                         group=patient, dicom=str(dicom), contour=str(contour)))
    with (out / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "mask", "group", "dicom", "contour"])
        writer.writeheader()
        writer.writerows(rows)
    return {"pairs": len(rows), "patients": len({r["group"] for r in rows}),
            "prepared_root": str(out), "target": "LV cavity (inner contour)",
            "normalization": "per-image min-max uint8"}
