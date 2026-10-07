from portable_bridge_unet.dataset_check import check_dataset

def test_missing_raw_dataset(tmp_path):
    report = check_dataset({"dataset": {
        "raw_root": str(tmp_path), "images": str(tmp_path / "images/*.png"),
        "masks": str(tmp_path / "masks/*.png")}})
    assert not report["raw_available"]
    assert not report["training_ready"]
    assert len(report["missing"]) == 7

def test_raw_inventory_does_not_claim_training_readiness(tmp_path):
    for i in range(1, 6):
        folder = tmp_path / f"SCD_IMAGES_{i:02d}"
        folder.mkdir()
        (folder / "sample.dcm").touch()
    contours = tmp_path / "SCD_ManualContours"
    contours.mkdir()
    (contours / "sample-icontour-manual.txt").write_text("1 1\n2 2\n")
    (tmp_path / "scd_patientdata.csv").touch()
    report = check_dataset({"dataset": {
        "raw_root": str(tmp_path), "images": str(tmp_path / "images/*.png"),
        "masks": str(tmp_path / "masks/*.png")}})
    assert report["raw_available"]
    assert not report["training_ready"]
    assert report["inner_contours"] == 1
