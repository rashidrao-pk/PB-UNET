# DRIVE: RGB retinal vessel segmentation

DRIVE contains 40 fundus images with an official 20-image training and 20-image
test split. Obtain the dataset through the
[official DRIVE site](https://drive.grand-challenge.org/), follow its access terms,
and retain the requested dataset citation. An account or access request may be
required; this repository does not automate authenticated downloads.

Extract the archives into the configured root:

```text
/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/raw/drive/DRIVE/
  training/
    images/21_training.tif
    1st_manual/21_manual1.gif
    mask/21_training_mask.gif
    ...
  test/
    images/01_test.tif
    1st_manual/01_manual1.gif
    mask/01_test_mask.gif
    ...
```

Point `dataset.raw_root` at the directory containing `training/` and `test/`.
The first manual annotation is required in both subsets; a package with only
unlabeled test images is insufficient for local evaluation.

```bash
python -m pip install -e .
python scripts/prepare_drive.py --config configs/drive_pb_epito.yaml --dry-run
python scripts/prepare_drive.py --config configs/drive_pb_epito.yaml
python scripts/check_dataset.py --config configs/drive_pb_epito.yaml --require-prepared
python scripts/train.py --config configs/drive_pb_epito.yaml
python scripts/evaluate.py --config configs/drive_pb_epito.yaml
python scripts/run_benchmark.py --config configs/benchmark_drive_epito.yaml
```

Use `drive_unet_epito.yaml` for U-Net. Local configs omit `_epito`. Library users
can call `data.preprocess.prepare_dataset(config)` or the unified `pb-data
preprocess --config ...` CLI. Preparation algorithms live in `data/preprocess/drive.py`.

## Reproducibility statement

The official 20 test images remain test-only. Seed 42 holds out 20% of the official
training images for validation (16 training, 4 validation, 20 test). RGB values
are retained in lossless PNGs. The first manual annotation defines vessels;
binary masks are intersected with the supplied retinal field of view (FOV).
FOV masks are retained separately in `fov/` and referenced by the `roi` column
of each split manifest. Source image, annotation, and FOV hashes and paths are
recorded. Source dimensions, nonempty binary masks, counts, disjoint IDs and
cross-subset byte duplicates are validated. Patient linkage is unavailable to
the preparer; this is an official image split, not a verified patient split.

No CLAHE, green-channel extraction, patch sampling, or morphological filtering
is used. During loading images are resized bilinearly to 256×256 and divided
by 255; target/FOV masks use nearest-neighbor resizing. Training loss remains
the existing full-image binary loss. Training/validation metrics and evaluation
confusion counts use **only pixels within FOV**, including accuracy's denominator.
HD95 compares FOV-clipped vessel masks in resized pixels. Largest-component
postprocessing is disabled because disconnected vessel branches are valid.
Do not compare this resized whole-image baseline directly with native-resolution,
patch-based or differently preprocessed published results.

Preparation requires an empty destination. Preserve `metadata/preparation.json`,
`manifest.csv` and all split CSVs with each experiment. Evaluation must use a
split CSV that retains `roi`; the generic PNG inventory alone does not carry FOV
metadata. Existing figures show image, binary vessel mask and vessel overlay.

Generate all new dataset previews and a combined figure after preparation:

```bash
python scripts/preview_datasets.py --configs configs/brats2020_pb_epito.yaml configs/brats2021_pb_epito.yaml configs/drive_pb_epito.yaml
python scripts/plot_dataset_overview.py --datasets brats2020 brats2021 drive
```
