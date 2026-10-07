import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def test_overview_all_datasets_and_source_manifest(tmp_path):
    from portable_bridge_unet.data import overview as module
    smoke = tmp_path / "smoke"
    for name in module.DATASETS:
        folder = smoke / name / "20260101"
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "sample_00.png"), np.zeros((32, 96, 3), np.uint8))
        (folder / "report.json").write_text(json.dumps(dict(created_utc="20260101",
            dataset=dict(dataset_name=name, prepared_pairs=20), samples=[dict(preview="sample_00.png")])) )
    selected = module.find_previews(smoke, 0)
    assert set(selected) == set(module.DATASETS)
    out = tmp_path / "overview"
    module.create_dataset_overview(smoke, out)
    assert (out / "dataset_overview.png").is_file()
    assert (out / "dataset_overview.pdf").is_file()
    manifest = json.loads((out / "dataset_overview_sources.json").read_text())
    assert "both lung fields" in manifest["montgomery"]["description"]
    assert "RGB dermoscopic" in manifest["isic2016"]["description"]
    with pytest.raises(ValueError, match="Missing saved previews"):
        module.find_previews(smoke, 1)


def test_download_listing_accepts_only_expected_filenames():
    from portable_bridge_unet.data.retrieve import montgomery as module
    links = module.FileLinks()
    links.feed('<a href="MCUCXR_0001_0.png">x</a><a href="../outside.png">x</a>'
               '<a href="MCUCXR_0001_0.png">duplicate</a>')
    assert links.names == {"MCUCXR_0001_0.png"}
