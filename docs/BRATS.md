# BraTS 2020 / 2021: 2D brain-tumor experiments

These integrations use the **labeled training releases** separately. They train
existing binary 2D models on single-modality axial MRI slices. Defaults are FLAIR
and whole tumor (WT = labels 1, 2, 4). They do not implement the multimodal 3D
challenge submission or volume-level challenge metrics.

## Obtain the data

Register and request access through the official release pages:

- [BraTS 2020 data and registration](https://www.med.upenn.edu/cbica/brats2020/data.html)
- [BraTS 2021 registration and data access](https://www.med.upenn.edu/cbica/brats2021/)

Follow the access instructions and citations on those pages. Downloads may require
an account and acceptance of terms; this repository does not embed credentials
or substitute an unofficial mirror. Use the original 2020/2021 labels (0,1,2,4),
not a newer release with enhancing tumor relabeled to 3.

Extract the labeled training archives under these Epito roots (nested grade or
archive folders are supported):

```text
/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/raw/brats2020/
  .../BraTS20_Training_001/
    BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_t1.nii.gz
    BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_001_seg.nii.gz
/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/raw/brats2021/
  .../BraTS2021_00000/
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_seg.nii.gz
    ...
```

Configs expect 369 labeled cases for 2020 and 1251 for 2021. Change `raw_root`
for your extraction location. For a deliberate subset, change `expected_cases`
and document why. Official unlabeled validation/test volumes cannot be used by
this supervised preparer. Never merge the two releases as independent cohorts:
subjects can recur across releases.

## Prepare and run on Epito

```bash
python -m pip install -e .
python scripts/prepare_brats2020.py --config configs/brats2020_pb_epito.yaml --dry-run
python scripts/prepare_brats2020.py --config configs/brats2020_pb_epito.yaml
python scripts/check_dataset.py --config configs/brats2020_pb_epito.yaml --require-prepared
python scripts/train.py --config configs/brats2020_pb_epito.yaml
python scripts/evaluate.py --config configs/brats2020_pb_epito.yaml
```

For 2021 replace `brats2020` with `brats2021`. PB and U-Net configs share the
prepared split files. All-model runners use `configs/benchmark_brats2020_epito.yaml`
and `configs/benchmark_brats2021_epito.yaml`. Local configs omit `_epito`.
Preparation can also be called through `python -m portable_bridge_unet.data
preprocess --config ...`. Algorithms live in `data/preprocess/brats.py`.

## Reproducibility statement

Volumes are reoriented to canonical RAS axes without interpolation. Image/mask
shapes and affines, finite values, and segmentation labels are checked. Seed 42
splits cases into approximately 70/15/15 percent training/validation/internal
test **before slicing**. All slices from a case remain in the same split.
Nonzero MRI voxels define each volume's 1st/99th intensity percentiles. Intensities
are clipped to that interval, scaled and rounded to uint8; original zero
background remains zero. Every axial slice containing nonzero MRI is exported,
including slices without tumor. MRI-empty slices are omitted. Original volume
hashes, canonical affines, percentile bounds, source paths, case assignments,
and slice indices are retained in metadata. No annotation-dependent slice
selection, cropping, or preparation-time resizing is used.

`dataset.modality` can select `flair`, `t1`, `t1ce`, or `t2`; `dataset.region`
can select WT (1,2,4), TC (1,4), or ET (4). Use a **separate prepared_root and
run directory** for each choice and update every image/mask/split path. The
supplied configs use one input channel and one binary output. No four-channel
fusion or simultaneous multiclass output is claimed.

Loading resizes images bilinearly and masks with nearest-neighbor interpolation
to 256×256; images are divided by 255. Supplied configs disable augmentation and
postprocessing. Existing evaluation reports per-slice binary metrics and HD95
in resized-image pixels, not 3D millimeters. Slice observations from one case
are correlated: image-level bootstrap and Wilcoxon outputs are descriptive and
do not establish independent patient-level significance. Use case/volume-level
analysis for clinical or challenge comparisons.

Preparation requires an empty destination to avoid stale slices or conflicting
protocols. Retain completed preparation metadata; use a new destination for a
changed protocol. NIfTI validation and conversion can take time and produce many
PNG files. `--dry-run` reads and validates volumes without writing files.
