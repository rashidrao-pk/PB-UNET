"""Data APIs work independently of the repository's command-line scripts."""
from io import BytesIO
import json

import cv2
import numpy as np
import pytest
import yaml

from portable_bridge_unet.data import Sample
from portable_bridge_unet.data.loaders import Sample as LoaderSample
from portable_bridge_unet.data.preprocess import prepare_dataset
from portable_bridge_unet.data.retrieve import download_dataset


def test_compatibility_imports_share_implementations():
    from portable_bridge_unet import preparation, kvasir, montgomery, isic2016, dataset_check, smoke_test
    from portable_bridge_unet.data.preprocess import sunnybrook, kvasir as new_kvasir
    from portable_bridge_unet.data.preprocess import montgomery as new_montgomery, isic2016 as new_isic
    from portable_bridge_unet.data import checks, visualization
    assert Sample is LoaderSample
    assert preparation.prepare_dataset is sunnybrook.prepare_dataset
    assert kvasir.prepare_kvasir is new_kvasir.prepare_kvasir
    assert montgomery.prepare_montgomery is new_montgomery.prepare_montgomery
    assert isic2016.prepare_isic2016 is new_isic.prepare_isic2016
    assert dataset_check.check_dataset is checks.check_dataset
    assert smoke_test.save_smoke_test is visualization.save_smoke_test


def test_library_retrieval_dry_run_does_not_write(tmp_path, monkeypatch):
    from portable_bridge_unet.data.retrieve import isic2016
    html = "".join(f'<a href="https://isic-archive.s3.amazonaws.com/challenges/2016/{folder}.zip">Download</a>'
                   for folder in isic2016.FOLDERS.values())
    monkeypatch.setattr(isic2016, "urlopen", lambda *args, **kwargs: BytesIO(html.encode()))
    root = tmp_path / "raw"
    config = {"dataset": {"name": "isic2016", "raw_root": str(root)}}
    report = download_dataset(config, dry_run=True)
    assert len(report["archives"]) == 4
    assert report["dry_run"]
    assert not root.exists()


def test_library_download_montgomery_offline(tmp_path, monkeypatch):
    from portable_bridge_unet.data.retrieve import montgomery
    filename = "MCUCXR_0001_0.png"
    image = np.full((16, 16), 128, np.uint8)
    _, encoded = cv2.imencode(".png", image)
    requests = []

    def response(url, **kwargs):
        requests.append(url)
        if url.endswith("index.html"):
            return BytesIO(f'<a href="{filename}">image</a>'.encode())
        return BytesIO(encoded.tobytes())

    monkeypatch.setattr(montgomery, "urlopen", response)
    config = {"dataset": {"name": "montgomery", "raw_root": str(tmp_path / "raw"), "expected_pairs": 1}}
    report = download_dataset(config)
    assert report["images"] == 1 and report["masks"] == 2
    assert len(requests) == 6
    assert len(list((tmp_path / "raw").rglob("*.png"))) == 3
    requests.clear()
    download_dataset(config)
    assert len(requests) == 3  # Existing readable images need only remote inventory checks.


def test_unified_cli_uses_library_api(tmp_path, monkeypatch, capsys):
    from portable_bridge_unet.data import cli
    from portable_bridge_unet.data.retrieve import isic2016
    html = "".join(f'<a href="https://isic-archive.s3.amazonaws.com/challenges/2016/{folder}.zip">Download</a>'
                   for folder in isic2016.FOLDERS.values())
    monkeypatch.setattr(isic2016, "urlopen", lambda *args, **kwargs: BytesIO(html.encode()))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"dataset": {"name": "isic2016", "raw_root": str(tmp_path / "raw")}}))
    assert cli.main(["retrieve", "--config", str(config), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"]
    with pytest.raises(SystemExit) as exc:
        cli.dataset_main("preprocess", "montgomery", "unused.yaml", ["--config", str(config), "--dry-run"])
    assert exc.value.code == 1


@pytest.mark.parametrize("function", [download_dataset, prepare_dataset])
def test_unimplemented_dataset_is_explicit(function):
    with pytest.raises(ValueError, match="supported"):
        function({"dataset": {"name": "unimplemented"}}, dry_run=True)
