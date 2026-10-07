import json
import cv2
import numpy as np
from portable_bridge_unet.data.visualization import save_smoke_test

def test_previews_and_model_forward(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for i in range(3):
        cv2.imwrite(str(images / f"{i}.png"), np.full((24, 24), 128, np.uint8))
        cv2.imwrite(str(masks / f"{i}.png"), np.pad(np.full((8, 8), 255, np.uint8), 8))
    config = dict(model="unet", image_size=32, channels=1, filters=[4, 8, 16],
                  dataset=dict(images=str(images / "*.png"), masks=str(masks / "*.png")))
    inventory = dict(raw_available=True, training_ready=True)
    out, report = save_smoke_test(config, inventory, tmp_path / "smoke", num_samples=2)
    assert report["pretraining_ready"]
    assert report["model"]["output_shape"] == [1, 1, 32, 32]
    assert len(report["samples"]) == 2
    preview = cv2.imread(str(out / report["samples"][0]["preview"]))
    assert preview.shape[0] > 32 and preview.shape[1] > 96
    assert "BaselineUNet" in (out / "model_architecture.txt").read_text()
    assert json.loads((out / "resolved_config.json").read_text())["channels"] == 1
    assert json.loads((out / "report.json").read_text())["pretraining_ready"]
    out2, _ = save_smoke_test(config, inventory, tmp_path / "smoke", 1)
    assert out2 != out and (out / "report.json").exists()

def test_missing_data_keeps_failure_artifacts(tmp_path):
    out, report = save_smoke_test(
        dict(model="unet", filters=[4, 8, 16]),
        dict(raw_available=False, training_ready=False), tmp_path)
    assert not report["pretraining_ready"]
    assert report["errors"]
    assert (out / "resolved_config.json").exists()
    assert (out / "model_architecture.txt").exists()
    assert not report["model"]["forward_pass"]
