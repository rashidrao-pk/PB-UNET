"""Reproducible Kvasir-SEG preparation: preserve RGB intensities, binarize masks."""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .data import pair_by_stem, split_samples_legacy, Sample

def source_pairs(config):
    dataset = config["dataset"]
    root = Path(dataset["raw_root"])
    extensions = {".jpg", ".jpeg", ".png"}
    files = lambda folder: sorted(str(p) for p in folder.iterdir()
                                  if p.is_file() and p.suffix.lower() in extensions)
    pairs = pair_by_stem(files(root / "images"), files(root / "masks"))
    expected = dataset.get("expected_pairs", 1000)
    if expected is not None and len(pairs) != expected:
        raise ValueError(f"expected {expected} Kvasir-SEG pairs, found {len(pairs)}")
    return pairs

def read_pair(sample):
    image = cv2.imread(sample.image, cv2.IMREAD_COLOR)
    mask = cv2.imread(sample.mask, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"unreadable image or mask: {sample}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image/mask dimensions differ: {sample.image}")
    return image, (mask > 127).astype(np.uint8) * 255

def prepare_kvasir(config, dry_run=False):
    pairs = source_pairs(config)
    # Validate the complete source set before producing prepared files.
    for pair in pairs:
        read_pair(pair)
    if dry_run:
        return {"pairs": len(pairs), "source_valid": True}
    out = Path(config["dataset"]["prepared_root"])
    for folder in ("images", "masks", "metadata"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    prepared, rows = [], []
    for pair in pairs:
        image, mask = read_pair(pair)
        name = Path(pair.image).stem + ".png"
        image_path, mask_path = out / "images" / name, out / "masks" / name
        for path, array in ((image_path, image), (mask_path, mask)):
            if not cv2.imwrite(str(path), array):
                raise OSError(f"failed to write {path}")
        prepared.append(Sample(str(image_path.resolve()), str(mask_path.resolve())))
        rows.append(dict(image=str(image_path.resolve()), mask=str(mask_path.resolve()),
                         source_image=pair.image, source_mask=pair.mask,
                         source_image_sha256=hashlib.sha256(Path(pair.image).read_bytes()).hexdigest(),
                         source_mask_sha256=hashlib.sha256(Path(pair.mask).read_bytes()).hexdigest(),
                         height=image.shape[0], width=image.shape[1],
                         foreground_pixels=int((mask > 0).sum())))
    with (out / "metadata/manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split = config.get("split", {})
    parts = split_samples_legacy(prepared, val_fraction=split.get("validation", .15),
                                 test_fraction=split.get("test", .15), seed=split.get("seed", 42))
    for name, samples in zip(("train", "val", "test"), parts):
        with (out / f"metadata/{name}_split.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "mask", "group"])
            writer.writerows((s.image, s.mask, "") for s in samples)
    report = dict(pairs=len(pairs), train=len(parts[0]), validation=len(parts[1]), test=len(parts[2]),
                  seed=split.get("seed", 42), split_unit="image; patient identities unavailable",
                  target="polyp foreground", mask_threshold=127,
                  preparation="decoded RGB retained without intensity normalization or resizing; PNG output",
                  prepared_root=str(out.resolve()))
    (out / "metadata/preparation.json").write_text(json.dumps(report, indent=2))
    return report
