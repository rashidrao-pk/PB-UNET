"""Read-only inventory of Sunnybrook downloads and prepared training pairs."""
from pathlib import Path
from glob import glob
from .data import pair_by_stem, SegmentationDataset

def check_dataset(config):
    dataset = config["dataset"]
    kind = dataset.get("name", "sunnybrook" if dataset.get("raw_root") else "prepared_pairs")
    if kind.lower() == "sunnybrook":
        root = Path(dataset["raw_root"])
        batches = [root / f"SCD_IMAGES_{i:02d}" for i in range(1, 6)]
        dicom_counts = {p.name: sum(1 for _ in p.rglob("*.dcm")) for p in batches}
        contours = root / "SCD_ManualContours"
        inner = sum(1 for _ in contours.rglob("*-icontour-manual.txt"))
        outer = sum(1 for _ in contours.rglob("*-ocontour-manual.txt"))
        missing = [str(p) for p in batches if not p.is_dir() or dicom_counts[p.name] == 0]
        if not inner:
            missing.append(str(contours / "*-icontour-manual.txt"))
        if not (root / "scd_patientdata.csv").is_file():
            missing.append(str(root / "scd_patientdata.csv"))
        report = dict(raw_root=str(root), raw_available=not missing, missing=missing,
                      dicom_counts=dicom_counts, inner_contours=inner, outer_contours=outer,
                      training_ready=False, prepared_pairs=0)
    else:
        root = Path(dataset["raw_root"]) if dataset.get("raw_root") else None
        report = dict(raw_root=str(root) if root else None,
                      raw_available=root.is_dir() if root else None,
                      missing=[], training_ready=False, prepared_pairs=0)
    if kind.lower() == "kvasir_seg":
        from .kvasir import source_pairs, read_pair
        try:
            raw_pairs = source_pairs(config)
            for pair in raw_pairs:
                read_pair(pair)
            report["raw_available"] = True
            report["raw_pairs"] = len(raw_pairs)
        except (OSError, ValueError) as exc:
            report["raw_available"] = False
            report["missing"].append(str(exc))
    report["dataset_name"] = kind
    if kind.lower() == "montgomery":
        from .montgomery import source_pairs, read_pair
        try:
            raw_pairs = source_pairs(config)
            for pair in raw_pairs:
                read_pair(pair)
            report.update(raw_available=True, raw_pairs=len(raw_pairs))
        except (OSError, ValueError) as exc:
            report["raw_available"] = False
            report["missing"].append(str(exc))
    try:
        samples = pair_by_stem(sorted(glob(dataset["images"], recursive=True)),
                               sorted(glob(dataset["masks"], recursive=True)))
        expected = dataset.get("expected_pairs")
        if expected is not None and len(samples) != expected:
            raise ValueError(f"expected {expected} prepared pairs, found {len(samples)}")
        ds = SegmentationDataset(samples, config.get("image_size", 256), config.get("channels", 3))
        for index in range(len(ds)):
            ds[index]
        report.update(training_ready=True, prepared_pairs=len(samples))
        if report["raw_available"] is None:
            report["raw_available"] = True
            report["raw_check"] = "not configured; checking prepared pairs only"
    except (ValueError, FileNotFoundError) as exc:
        report["preparation_issue"] = str(exc)
    return report
