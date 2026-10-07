# Montgomery County: lung segmentation

Montgomery adds lung anatomy in posterior-anterior chest X-rays to the existing
cardiac MRI, breast ultrasound, and colonoscopy experiments. The
[NLM dataset page](https://www.lhncbc.nlm.nih.gov/LHC-publications/pubs/TuberculosisChestXrayDatasets.html)
describes 138 images (80 normal and 58 abnormal with TB-consistent findings),
with separate manual left- and right-lung masks. We retain both categories and
segment their **lung fields**, not TB lesions. This is a small exploratory
benchmark, not a large clinical validation cohort.

Review the [official readme and citation/use instructions](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf)
before using or distributing data or derived figures. Follow NLM attribution
and citation requirements; the repository does not redistribute the dataset.

## Download on Epito

Activate the same PyTorch environment used for the other datasets, then run from
the repo root. Dataset paths are taken from the selected YAML:

```bash
python scripts/download_montgomery.py --config configs/montgomery_pb_epito.yaml --dry-run
python scripts/download_montgomery.py --config configs/montgomery_pb_epito.yaml
```

The first command checks official remote listings without downloading images.
The second downloads X-rays and both masks (414 PNG files) with a progress bar,
skips completed readable images on rerun, and replaces partial downloads only
after successful decoding. This requires internet access on the execution node;
if unavailable, run on a connected login node sharing the same storage.
The script downloads segmentation inputs only, not clinical-reading files or
TB-consensus annotation CSVs. The official readme remains linked above.

Configured source layout:

```text
/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/raw/montgomery/MontgomerySet/
  CXR_png/MCUCXR_0001_0.png
  ManualMask/leftMask/MCUCXR_0001_0.png
  ManualMask/rightMask/MCUCXR_0001_0.png
```

If downloaded elsewhere, change `dataset.raw_root` in the YAML. Point it at the
directory containing `CXR_png` and `ManualMask`, not its parent. Local configs
`montgomery_pb.yaml` and `montgomery_unet.yaml` use the existing macOS data root;
all `_epito` configs use `/beegfs`. No download has been performed on Epito by
this code change.

## Prepare and inspect

```bash
python scripts/prepare_montgomery.py --config configs/montgomery_pb_epito.yaml --dry-run
python scripts/prepare_montgomery.py --config configs/montgomery_pb_epito.yaml
python scripts/check_dataset.py --config configs/montgomery_pb_epito.yaml --require-prepared
```

Preparation checks exact image/left-mask/right-mask filename coverage, expected
count, decoding, dimensions, binary nonempty masks, and unique numeric filename
IDs. It writes prepared images, combined masks, SHA-256 source manifests,
preparation metadata, and explicit split CSVs under
`/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/montgomery`.

The split is stratified by the filename's normal/abnormal suffix, seed 42:
**98 training / 20 validation / 20 test** images. Validation and test sizes are
`floor(138 * 0.15)` each; training receives the remainder. Numeric filename IDs
are recorded as `group` and cannot repeat. This prevents filename-ID overlap;
independent patient linkage is not available to this implementation, so do not
claim a separately verified patient-level split. All compared models reuse the
same prepared manifests. Repreparation regenerates the same splits with unchanged
source files and config; retain manifests for completed experiments.

The checker produces labeled image/mask/overlay previews, architecture text,
resolved settings, and readiness reports in
`runs/smoke_test/montgomery/<timestamp>/`. A missing or invalid dataset fails the
check rather than generating substitute data.

## Preprocessing statement

> Original chest radiographs were decoded as grayscale without resizing during
> preparation. Uint8 images retained their original intensities; uint16 images
> were independently min–max mapped to uint8 using rounded values in [0,255]
> (constant images map to zero). Source dtype, range, paths, and SHA-256 hashes
> were recorded. Left- and right-lung binary annotations were combined by logical
> union of nonzero pixels and saved as lossless 0/255 PNG masks. During loading,
> images were resized to 256×256 using bilinear interpolation, masks using
> nearest-neighbor interpolation, and image values divided by 255. The default
> three-channel input repeats grayscale through color decoding. Augmentation
> and postprocessing were disabled in the supplied experiment configs.

The readme describes 12-bit acquisition; reading with `IMREAD_UNCHANGED` avoids
silently truncating high-bit-depth PNGs. The normal/abnormal labels are used only
for stratification, not as segmentation targets. Image resizing is square, so
aspect ratio is not retained; HD95 is measured in resized-image pixels, not mm.

## Train and evaluate matched models

```bash
python scripts/train.py --config configs/montgomery_pb_epito.yaml
python scripts/train.py --config configs/montgomery_unet_epito.yaml
python scripts/evaluate.py --config configs/montgomery_pb_epito.yaml
python scripts/evaluate.py --config configs/montgomery_unet_epito.yaml
python scripts/compare_models.py --config configs/comparison_montgomery.yaml
python scripts/bootstrap_comparison.py --config configs/comparison_montgomery.yaml
```

Both models use BCE–Dice, identical splits/settings, and `best_dice.pt` selection.
Training/evaluation plots are generated by the existing scripts. For a prediction,
set `predict.image` in the YAML or pass `--image path/to/prepared/image.png`:

```bash
python scripts/predict.py --config configs/montgomery_pb_epito.yaml --image PATH_TO_PREPARED_IMAGE
```

To benchmark every implemented architecture with progress bars:

```bash
python scripts/run_benchmark.py --config configs/benchmark_montgomery_epito.yaml --dry-run
python scripts/run_benchmark.py --config configs/benchmark_montgomery_epito.yaml
```

Benchmark outputs are separate under `runs/benchmark_montgomery`. The standalone
comparison config above refers to `runs/montgomery_pb` and `runs/montgomery_unet`;
the benchmark already generates its own paired comparisons and bootstrap output.

**Keep postprocessing disabled:** the existing hole-fill/largest-component
pipeline can discard an entire lung. The benchmark config sets `postprocess:
false`; standalone configs also default to raw outputs. Passing `--postprocess`
explicitly overrides that choice and should not be used for the primary lung
comparison. No lung-specific postprocessor is implemented here.

## Update the dataset figures

```bash
python scripts/preview_datasets.py --configs configs/montgomery_pb_epito.yaml
python scripts/plot_dataset_overview.py
```

The overview now includes Montgomery as a fourth row with its organ/modality
description. To retain a three-dataset figure before downloading Montgomery:

```bash
python scripts/plot_dataset_overview.py --datasets sunnybrook busi kvasir_seg
```

Preparation tests use synthetic files to verify both-lung retention, high-bit
depth handling, deterministic disjoint stratified splits, provenance, smoke
figures, and rejection of corrupt/incomplete source layouts. Actual dataset
availability, download completion, and model performance must be checked on
Epito; no trained Montgomery results are claimed by this integration.
