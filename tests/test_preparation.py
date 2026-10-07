import cv2
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from portable_bridge_unet.data.preprocess.sunnybrook import prepare_dataset, plan_samples, rasterize_contour

def test_prepare_and_manifest(tmp_path):
    raw = tmp_path / "raw"
    series = raw / "SCD_IMAGES_01/SCD0000101/CINESAX_300"
    series.mkdir(parents=True)
    (raw / "scd_patientdata.csv").write_text("\ufeffPatientID,OriginalID\nSCD0000101,SC-HF-I-1\n")
    folder = raw / "SCD_ManualContours/SC-HF-I-01/contours-manual/IRCCI-expert"
    folder.mkdir(parents=True)
    contour = folder / "IM-0001-0001-icontour-manual.txt"
    contour.write_text("2 2\n5 2\n5 5\n2 5\n")
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    path = series / "IM-0003-0001.dcm"
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = "SCD0000101"
    ds.InstanceNumber = 1
    ds.Rows = ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.arange(64, dtype=np.uint16).reshape(8, 8).tobytes()
    ds.save_as(path, enforce_file_format=True)
    out = tmp_path / "prepared"
    config = {"dataset": {"raw_root": str(raw), "prepared_root": str(out)}}
    report = prepare_dataset(config)
    assert report["pairs"] == report["patients"] == 1
    image = cv2.imread(str(out / "images/SCD0000101_0001.png"), 0)
    mask = cv2.imread(str(out / "masks/SCD0000101_0001.png"), 0)
    assert image.min() == 0 and image.max() == 255
    assert mask[3, 3] == 255 and mask[0, 0] == 0
    assert "SCD0000101" in (out / "manifest.csv").read_text()
    duplicate = series.parent / "CINESAX_301"
    duplicate.mkdir()
    (duplicate / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="found 2"):
        plan_samples(config)
    config["dataset"]["series_overrides"] = {"SCD0000101": "CINESAX_300"}
    assert len(plan_samples(config)) == 1

def test_out_of_bounds_contour(tmp_path):
    path = tmp_path / "contour.txt"
    path.write_text("0 0\n100 0\n2 2\n")
    with pytest.raises(ValueError, match="bounds"):
        rasterize_contour(path, (8, 8))
