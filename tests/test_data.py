import cv2
import numpy as np
import pytest
import torch
from portable_bridge_unet.data import Sample, SegmentationDataset, pair_by_stem, split_samples_grouped, split_samples_legacy

@pytest.mark.parametrize("channels", [1, 3])
def test_loading_and_batch(tmp_path, channels):
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    cv2.imwrite(str(image), np.full((10, 10, 3), (0, 0, 255), dtype=np.uint8))
    cv2.imwrite(str(mask), np.pad(np.full((4, 4), 255, dtype=np.uint8), 3))
    item = SegmentationDataset([Sample(str(image), str(mask))], 16, channels)[0]
    assert item["image"].shape == (channels, 16, 16)
    assert item["mask"].shape == (1, 16, 16)
    assert item["image"].dtype == torch.float32
    assert set(item["mask"].unique().tolist()) == {0., 1.}
    if channels == 3:
        assert item["image"][0].min() == 1
        assert item["image"][2].max() == 0

def test_pairing_errors():
    assert pair_by_stem(["a.png", "b.png"], ["b.jpg", "a.jpg"])[0].image == "a.png"
    with pytest.raises(ValueError):
        pair_by_stem(["a.png", "b.png"], ["a.png"])
    with pytest.raises(ValueError):
        pair_by_stem([], [])

def test_splits_are_reproducible_and_disjoint():
    samples = [Sample(str(i), str(i), str(i // 2)) for i in range(100)]
    for splitter in (split_samples_grouped, split_samples_legacy):
        parts = splitter(samples)
        assert parts == splitter(samples)
        assert sum(map(len, parts)) == len(samples)
        sets = [set(p) for p in parts]
        assert not (sets[0] & sets[1] or sets[1] & sets[2] or sets[0] & sets[2])
    groups = [{s.group for s in p} for p in split_samples_grouped(samples)]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])

def test_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        SegmentationDataset([Sample(str(tmp_path / "missing.png"), "missing.png")])[0]

def test_duplicate_stems_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        pair_by_stem(["patient1/a.png", "patient2/a.png"], ["a.png"])
