import csv
import cv2
import numpy as np
import pytest
from portable_bridge_unet.kvasir import prepare_kvasir, source_pairs
from portable_bridge_unet.dataset_check import check_dataset

def fixture_config(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "prepared"
    for name in ("images", "masks"):
        (raw / name).mkdir(parents=True)
    for i in range(20):
        image = np.zeros((12, 18, 3), np.uint8)
        image[..., 2] = 200
        mask = np.zeros((12, 18), np.uint8)
        mask[3:8, 4:10] = 255
        cv2.imwrite(str(raw / f"images/{i:02d}.jpg"), image)
        cv2.imwrite(str(raw / f"masks/{i:02d}.jpg"), mask)
    return dict(dataset=dict(name="kvasir_seg", raw_root=str(raw), prepared_root=str(out),
                            images=str(out / "images/*.png"), masks=str(out / "masks/*.png"),
                            expected_pairs=20), split=dict(seed=42, validation=.15, test=.15))

def test_preparation_color_binary_and_shared_splits(tmp_path):
    config = fixture_config(tmp_path)
    assert prepare_kvasir(config, True)["source_valid"]
    out = tmp_path / "prepared"
    assert not out.exists()
    report = prepare_kvasir(config)
    assert (report["train"], report["validation"], report["test"]) == (14, 3, 3)
    image = cv2.imread(str(out / "images/00.png"))
    assert image[0, 0, 2] > 190 and image[0, 0, 0] < 5
    assert set(np.unique(cv2.imread(str(out / "masks/00.png"), 0))) == {0, 255}
    manifest = list(csv.DictReader((out / "metadata/manifest.csv").open()))
    assert len(manifest[0]["source_image_sha256"]) == 64
    splits = [list(csv.DictReader((out / f"metadata/{n}_split.csv").open())) for n in ("train", "val", "test")]
    identities = [{r["image"] for r in part} for part in splits]
    assert not (identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2])
    before = (out / "metadata/test_split.csv").read_text()
    prepare_kvasir(config)
    assert before == (out / "metadata/test_split.csv").read_text()
    assert check_dataset(config)["training_ready"]

def test_wrong_count_and_dimensions(tmp_path):
    config = fixture_config(tmp_path)
    config["dataset"]["expected_pairs"] = 1000
    with pytest.raises(ValueError, match="expected 1000"):
        source_pairs(config)
    config["dataset"]["expected_pairs"] = 20
    cv2.imwrite(str(tmp_path / "raw/masks/00.jpg"), np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="dimensions differ"):
        prepare_kvasir(config)
    assert not check_dataset(config)["raw_available"]
