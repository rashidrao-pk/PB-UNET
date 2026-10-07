import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overview_four_rows_and_source_manifest(tmp_path, monkeypatch):
    module = load_script("plot_dataset_overview")
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
    monkeypatch.setattr(sys, "argv", ["overview", "--smoke-dir", str(smoke), "--out-dir", str(out)])
    module.main()
    assert (out / "dataset_overview.png").is_file()
    assert (out / "dataset_overview.pdf").is_file()
    manifest = json.loads((out / "dataset_overview_sources.json").read_text())
    assert "both lung fields" in manifest["montgomery"]["description"]
    with pytest.raises(ValueError, match="Missing saved previews"):
        module.find_previews(smoke, 1)


def test_download_listing_accepts_only_expected_filenames():
    module = load_script("download_montgomery")
    links = module.FileLinks()
    links.feed('<a href="MCUCXR_0001_0.png">x</a><a href="../outside.png">x</a>'
               '<a href="MCUCXR_0001_0.png">duplicate</a>')
    assert links.names == {"MCUCXR_0001_0.png"}
