import argparse
import pytest
from portable_bridge_unet.config import load_config, parse_config_args

def test_paths_and_cli_precedence(tmp_path):
    path = tmp_path / "test.yaml"
    path.write_text("model: unet\nlearning_rate: 0.01\ndataset:\n  images: images/*.png\n  masks: masks/*.png\ntrain:\n  epochs: 3\n")
    config = load_config(path)
    assert config["dataset"]["images"] == str(tmp_path / "images/*.png")
    parser = argparse.ArgumentParser()
    for key in ("images", "masks", "model", "lr"):
        parser.add_argument("--" + key)
    parser.add_argument("--epochs", type=int, default=100)
    args = parse_config_args(parser, "train", ["--config", str(path), "--epochs", "2"])
    assert args.epochs == 2
    assert args.lr == 0.01
    assert args.model == "unet"
    assert args.images == config["dataset"]["images"]

def test_invalid_config(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- not a mapping")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
