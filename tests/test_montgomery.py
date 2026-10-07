import csv
import json

import cv2
import numpy as np
import pytest

from portable_bridge_unet.dataset_check import check_dataset
from portable_bridge_unet.montgomery import prepare_montgomery, source_pairs
from portable_bridge_unet.smoke_test import save_smoke_test


def fixture_config(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "prepared"
    for folder in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"):
        (raw / folder).mkdir(parents=True)
    for i in range(20):
        name = f"MCUCXR_{i:04d}_{i % 2}.png"
        image = np.full((32, 32), 100 + i, np.uint8)
        left, right = np.zeros_like(image), np.zeros_like(image)
        left[4:28, 3:12] = 255
        right[4:28, 20:29] = 255
        for folder, array in zip(("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"),
                                 (image, left, right)):
            cv2.imwrite(str(raw / folder / name), array)
    return dict(model="unet", filters=[4, 8, 16], image_size=32, channels=3,
                dataset=dict(name="montgomery", raw_root=str(raw), prepared_root=str(out),
                             expected_pairs=20, images=str(out / "images/*.png"), masks=str(out / "masks/*.png")),
                split=dict(seed=42, validation=.15, test=.15))


def test_preparation_union_splits_and_smoke(tmp_path):
    config = fixture_config(tmp_path)
    assert prepare_montgomery(config, True)["source_valid"]
    out = tmp_path / "prepared"
    assert not out.exists()
    report = prepare_montgomery(config)
    assert (report["train"], report["validation"], report["test"]) == (14, 3, 3)
    image = cv2.imread(str(out / "images/MCUCXR_0000_0.png"), 0)
    assert np.all(image == 100)
    mask = cv2.imread(str(out / "masks/MCUCXR_0000_0.png"), 0)
    assert mask[10, 5] == mask[10, 22] == 255
    assert mask[10, 15] == 0
    assert set(np.unique(mask)) == {0, 255}
    splits = [list(csv.DictReader((out / f"metadata/{name}_split.csv").open()))
              for name in ("train", "val", "test")]
    groups = [{row["group"] for row in part} for part in splits]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    assert all({row["abnormal"] for row in part} == {"0", "1"} for part in splits)
    before = (out / "metadata/test_split.csv").read_text()
    prepare_montgomery(config)
    assert before == (out / "metadata/test_split.csv").read_text()
    manifest = list(csv.DictReader((out / "metadata/manifest.csv").open()))
    assert len(manifest[0]["source_right_mask_sha256"]) == 64
    inventory = check_dataset(config)
    assert inventory["raw_available"] and inventory["training_ready"]
    artifacts, smoke = save_smoke_test(config, inventory, tmp_path / "smoke", 2)
    assert smoke["pretraining_ready"]
    assert (artifacts / "sample_00.png").exists()
    assert json.loads((out / "metadata/preparation.json").read_text())["postprocessing"].startswith("disabled")


@pytest.mark.parametrize("failure", ["missing_mask", "dimensions", "empty", "count", "duplicate_patient"])
def test_reject_invalid_source(tmp_path, failure):
    config = fixture_config(tmp_path)
    path = tmp_path / "raw/ManualMask/rightMask/MCUCXR_0000_0.png"
    if failure == "missing_mask":
        path.unlink()
    elif failure == "dimensions":
        cv2.imwrite(str(path), np.full((8, 8), 255, np.uint8))
    elif failure == "empty":
        cv2.imwrite(str(path), np.zeros((32, 32), np.uint8))
    elif failure == "count":
        config["dataset"]["expected_pairs"] = 138
    else:
        for folder in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"):
            source = tmp_path / "raw" / folder / "MCUCXR_0001_1.png"
            source.rename(source.with_name("MCUCXR_0000_1.png"))
    with pytest.raises(ValueError):
        prepare_montgomery(config)
    assert not (tmp_path / "prepared").exists()
    assert not check_dataset(config)["raw_available"]


def test_high_bit_depth_is_explicitly_normalized(tmp_path):
    from portable_bridge_unet.montgomery import read_pair
    config = fixture_config(tmp_path)
    pair = source_pairs(config)[0]
    image = np.linspace(0, 4095, 32 * 32).reshape(32, 32).astype(np.uint16)
    cv2.imwrite(str(pair[0]), image)
    prepared, mask = read_pair(pair)
    assert prepared.dtype == np.uint8
    assert prepared.min() == 0 and prepared.max() == 255
    assert mask[10, 5] == mask[10, 22] == 255
