import csv
import importlib.util
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from portable_bridge_unet.dataset_check import check_dataset
from portable_bridge_unet.isic2016 import FOLDERS, prepare_isic2016, source_pairs
from portable_bridge_unet.smoke_test import save_smoke_test


def fixture_config(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "prepared"
    for folder in FOLDERS.values():
        (raw / folder).mkdir(parents=True)
    for part, indices in (("train", range(20)), ("test", range(100, 105))):
        for i in indices:
            image = np.zeros((32, 40, 3), np.uint8)
            image[..., 2] = 100 + i
            mask = np.zeros((32, 40), np.uint8)
            mask[8:24, 10:30] = 255
            cv2.imwrite(str(raw / FOLDERS[f"{part}_images"] / f"ISIC_{i:07d}.jpg"), image)
            cv2.imwrite(str(raw / FOLDERS[f"{part}_masks"] / f"ISIC_{i:07d}_Segmentation.png"), mask)
    return dict(model="unet", filters=[4, 8, 16], channels=3, image_size=32,
                dataset=dict(name="isic2016", raw_root=str(raw), prepared_root=str(out),
                             images=str(out / "images/*.png"), masks=str(out / "masks/*.png"),
                             expected_train_pairs=20, expected_test_pairs=5, expected_pairs=25),
                split=dict(validation=.2, seed=42))


def test_rgb_official_test_isolation_and_repeatability(tmp_path):
    config = fixture_config(tmp_path)
    assert prepare_isic2016(config, True)["source_valid"]
    out = tmp_path / "prepared"
    assert not out.exists()
    report = prepare_isic2016(config)
    assert (report["train"], report["validation"], report["test"]) == (16, 4, 5)
    splits = [list(csv.DictReader((out / f"metadata/{name}_split.csv").open()))
              for name in ("train", "val", "test")]
    ids = [{Path(row["image"]).stem for row in part} for part in splits]
    official_test = {Path(s.image).stem for s in source_pairs(config)["test"]}
    assert ids[2] == official_test
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    image = cv2.imread(str(out / "images/ISIC_0000100.png"))
    assert image[0, 0, 2] > 190 and image[0, 0, 0] < 5
    mask = cv2.imread(str(out / "masks/ISIC_0000100.png"), 0)
    assert set(np.unique(mask)) == {0, 255}
    before = (out / "metadata/test_split.csv").read_text()
    prepare_isic2016(config)
    assert before == (out / "metadata/test_split.csv").read_text()
    manifest = list(csv.DictReader((out / "metadata/manifest.csv").open()))
    assert len(manifest[0]["source_mask_sha256"]) == 64
    assert all(row["official_split"] == "test" for row in manifest if row["split"] == "test")
    inventory = check_dataset(config)
    assert inventory["raw_available"] and inventory["training_ready"]
    _, smoke = save_smoke_test(config, inventory, tmp_path / "smoke", 2)
    assert smoke["pretraining_ready"]
    assert smoke["samples"][0]["image_shape"] == [3, 32, 32]


@pytest.mark.parametrize("failure", ["count", "missing", "dimensions", "empty", "overlap", "duplicate_bytes"])
def test_invalid_isic_sources_fail_before_writes(tmp_path, failure):
    config = fixture_config(tmp_path)
    mask = tmp_path / "raw" / FOLDERS["train_masks"] / "ISIC_0000000_Segmentation.png"
    if failure == "count":
        config["dataset"]["expected_train_pairs"] = 900
    elif failure == "missing":
        mask.unlink()
    elif failure == "dimensions":
        cv2.imwrite(str(mask), np.full((8, 8), 255, np.uint8))
    elif failure == "empty":
        cv2.imwrite(str(mask), np.zeros((32, 40), np.uint8))
    elif failure == "overlap":
        for role, suffix in (("images", ".jpg"), ("masks", "_Segmentation.png")):
            path = tmp_path / "raw" / FOLDERS[f"test_{role}"] / ("ISIC_0000100" + suffix)
            path.rename(path.with_name("ISIC_0000000" + suffix))
    else:
        image = tmp_path / "raw" / FOLDERS["train_images"] / "ISIC_0000000.jpg"
        duplicate = tmp_path / "raw" / FOLDERS["test_images"] / "ISIC_0000100.jpg"
        duplicate.write_bytes(image.read_bytes())
    with pytest.raises(ValueError):
        prepare_isic2016(config)
    assert not (tmp_path / "prepared").exists()
    assert not check_dataset(config)["raw_available"]


def test_official_archive_links_and_safe_extraction(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/download_isic2016.py"
    spec = importlib.util.spec_from_file_location("download_isic2016", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    links = module.ArchiveLinks()
    for folder in FOLDERS.values():
        links.feed(f'<a href="https://isic-archive.s3.amazonaws.com/2016/{folder}.zip">Download</a>')
    assert len(links.links) == 4
    archive = tmp_path / "masks.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("nested/ISIC_0000001_Segmentation.png", b"test bytes")
        stream.writestr("../../outside.txt", b"skip")
    module.extract_images(archive, tmp_path / "extracted", masks=True)
    assert (tmp_path / "extracted/ISIC_0000001_Segmentation.png").read_bytes() == b"test bytes"
    assert not (tmp_path / "outside.txt").exists()
