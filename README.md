# Portable-Bridge U-Net — PyTorch edition

This is the **active PyTorch rewrite** of the supplied Portable-Bridge U-Net project. TensorFlow/Keras is no longer required for training, evaluation, inference, metrics, or post-processing.

The `portable_bridge_legacy` model preserves the paper-era architecture. The revised `portable_bridge` adds refinement, bilinear upsampling, and spatial dropout; it is a separate experiment.

## What was converted

| Old TensorFlow/Keras component   | PyTorch replacement                                      |
| -------------------------------- | -------------------------------------------------------- |
| `model.py` baseline U-Net        | `src/portable_bridge_unet/models.py::BaselineUNet`       |
| `prop_model.py` proposed network | `src/portable_bridge_unet/models.py::PortableBridgeLegacyUNet` |
| `tf.data` image pipeline         | `torch.utils.data.Dataset/DataLoader`                    |
| Keras BCE                        | `torch.nn.BCEWithLogitsLoss`                             |
| Keras Adam                       | `torch.optim.Adam`                                       |
| `ModelCheckpoint`                | PyTorch checkpoint `best.pt`                             |
| `ReduceLROnPlateau`              | `torch.optim.lr_scheduler.ReduceLROnPlateau`             |
| Keras training loop              | `scripts/train.py`                                       |
| prediction/evaluation notebooks  | `scripts/evaluate.py`, `scripts/predict.py`              |
| skimage post-processing          | OpenCV hole fill + largest connected component           |

The three Keras source files retained under `legacy_tensorflow_reference/` are **reference only** and are not imported by the new pipeline.

## Architecture parity

The most useful parity check is the number of trainable parameters:

```text
Baseline U-Net        PyTorch: 1,211,649
Keras manuscript trainable:   1,211,649

Portable-Bridge U-Net PyTorch: 2,356,609
Keras manuscript trainable:   2,356,609
```

Run:

```bash
python audit/verify_parameter_counts.py
```

The manuscript's larger Keras `total` counts (1,213,953 and 2,360,321) include BatchNorm running mean/variance as non-trainable values. PyTorch stores these as buffers rather than parameters, so they do not appear in `sum(model.parameters())`.

## Important architecture decision

The original Keras code mutates the filter list using `num_filters.reverse()`. Repeated model creation can therefore alternate between encoder orders such as:

```text
32 -> 64 -> 128
```

and

```text
128 -> 64 -> 32
```

The new implementation **never mutates the filter list**. The canonical paper configuration is fixed at:

```text
[32, 64, 128]
```

Some saved historical `.h5` checkpoints in the supplied archive were created after the list had already been reversed (their first convolution is `3 -> 128`). Those weights should **not** be silently loaded into the canonical PyTorch model. For publication-quality experiments, retrain with this deterministic implementation.

## Installation

```bash
cd portable_bridge_pytorch
python -m venv .venv
source .venv/bin/activate
# pip install -e .
pip install -e . --no-deps

pip install pytest
```

For an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build for your machine before `pip install -e .` if needed.

## Train the exact paper-era architecture

The implementation defaults to the code's actual 3-channel input, 256x256 images, BCE, Adam and filters 32/64/128:

```bash
python scripts/train.py --config configs/paper_legacy.yaml
```

Baseline:

```bash
python scripts/train.py --config configs/baseline.yaml
```

Prepare the data with `python scripts/prepare_sunnybrook.py`, then check readiness
with `python scripts/check_dataset.py --require-prepared`.
Dataset paths and hyperparameters live in the selected YAML file.

The trainer saves the **actual split files** used for the run:

```text
train_split.csv
val_split.csv
test_split.csv
best.pt
history.json
config.json
```

That makes the experiment reproducible and avoids the old notebook ambiguity.

## Evaluate the complete test set

Raw output:

```bash
python scripts/evaluate.py --config configs/paper_legacy.yaml
```

With the manuscript post-processing:

```bash
python scripts/evaluate.py --config configs/paper_legacy.yaml --postprocess
```

Use `configs/baseline.yaml` to evaluate the baseline run.

The evaluator reports **every image**, without the legacy script's score-based filtering or duplicate appending. Outputs include per-image CSV and mean/std/median for Dice, IoU, precision, recall, accuracy and HD95.

## Predict one image

```bash
python scripts/predict.py --config configs/paper_legacy.yaml --postprocess
```

Set `predict.image` in the YAML to the prepared image to predict. The checkpoint
and output paths are already configured. You can also override individual
settings on the command line.


## CPU / CUDA / Apple Silicon

Device is selected automatically:

```text
CUDA -> NVIDIA GPU
MPS  -> Apple Silicon GPU
CPU  -> fallback
```

Override with:

```bash
--device cuda:0
--device mps
--device cpu
```

Automatic mixed precision is enabled on CUDA by default. Disable with `--no-amp`.

## Data splitting

`scripts/train.py` currently defaults to the **legacy image-level 70/15/15 split** so that E0 can reproduce the old experiment as closely as possible.

The library also contains `split_samples_grouped(...)` for a patient-level split. For the revised publication, the recommended sequence is:

1. E0-Legacy: reproduce the original image-level experiment.
2. E0-Patient: rerun with patient-wise splitting after deriving patient IDs.
3. E0-Ablation: U-Net vs PB-U-Net vs PB-U-Net + post-processing.
4. E1+: add BUSI/Kvasir/other external datasets using the same PyTorch codebase.

## Input channels

The paper text says single-channel MRI, but the executable Keras implementation actually used `(256,256,3)` and `cv2.IMREAD_COLOR`. Therefore the parity configuration is `--channels 3`.

For a scientifically cleaner grayscale MRI rerun, use:

```bash
--channels 1
```

That is a new experiment and should not be described as an exact reproduction of the old implementation.

## Tests

```bash
PYTHONPATH=src pytest -q
```

Tests verify all model output dimensions, legacy parameter parity, finite training gradients and losses, checkpoint round trips, and data/statistical workflows.

## Recommended next step

Do **not** convert the questionable historical H5 weights blindly. Re-run the baseline and proposed models from scratch with this deterministic PyTorch implementation, save the complete test-set metrics, and use those results as the trusted foundation for the revised manuscript and external-dataset experiments.

## Shared configuration and dataset checks

All active scripts load `configs/paper_legacy.yaml` by default. Use
`--config path/to/experiment.yaml` to select another file; explicit CLI options
override its values. Relative paths in YAML resolve against the YAML directory.
The raw dataset path is defined once in `dataset.raw_root`.

```bash
python scripts/check_dataset.py
python scripts/prepare_sunnybrook.py
python scripts/check_dataset.py --require-prepared
python scripts/train.py --config configs/paper_legacy.yaml
python scripts/evaluate.py --config configs/paper_legacy.yaml
python scripts/predict.py --config configs/paper_legacy.yaml  # after setting predict.image
python -m pytest -q
```

The dataset checker inventories all five Sunnybrook DICOM batches, manual
inner/outer contours, and patient metadata. It also decodes every configured
prepared image/mask pair. The default exit status checks raw availability;
`--require-prepared` checks training readiness.

The downloaded Sunnybrook data contains DICOM files and contour coordinates,
not the PNG image/mask pairs expected by the current training loader. Run `python scripts/prepare_sunnybrook.py` to match patient/series identifiers
and rasterize inner contours as LV cavity masks. Store prepared files under
`data/sunnybrook/images` and `data/sunnybrook/masks`, or change their config
globs. Use unique patient-prefixed filenames to prevent pairing collisions.

Tests cover configuration overrides, image/mask loading and binary masks,
reproducible splits and patient separation, both model architectures, finite
training gradients, checkpoint prediction round trips, invalid checkpoints,
and raw dataset inventory. They use temporary fixtures and do not require the
local medical dataset or pretrained weights.

Dataset checks also save timestamped pre-training artifacts under
`runs/smoke_test`: image/mask/overlay previews, model architecture, resolved
configuration, and a JSON report with preprocessing and a CPU forward-pass
check. Run `python scripts/check_dataset.py --require-prepared` and inspect
these artifacts before training. See [the training guide](docs/TRAIN.md).

Training saves updated metric curves (`training_curves.png/.pdf`) each epoch.
Evaluation saves score distributions and summary plots in PNG/PDF, with
separate raw/postprocessed filenames. To generate plots for existing results,
run `python scripts/plot_results.py`.

For paired model statistics and reproducible bootstrap confidence intervals:

```bash
python scripts/compare_models.py --config configs/comparison_busi.yaml
python scripts/bootstrap_comparison.py --config configs/comparison_busi.yaml
```

Use `configs/comparison_sunnybrook.yaml` for existing Sunnybrook results.
See [evaluation documentation](docs/EVALUATE.md) for output files and interpretation.

For the next dataset, follow the [Kvasir-SEG guide](docs/KVASIR.md): official
download, preparation, labeled smoke-test previews, matched baseline/PB-U-Net
training, evaluation plots, and paired statistical comparisons.

## Revised PB-U-Net and architecture benchmark suite

The repository now keeps the exact paper-era model as `portable_bridge_legacy` and adds a revised `portable_bridge` (PB-U-Net-R) with bilinear feature alignment and a final decoder refinement block. An additional `apb_unet` variant adds lightweight channel gating before bridge skip fusion.

Available comparison architectures:

```text
unet
portable_bridge_legacy
portable_bridge
apb_unet
unetpp
attention_unet
resunet
resunetpp
unet3plus
```

The trainer also supports `--loss bce_dice`, which is recommended for revised segmentation experiments. Use exactly the same loss and split for all architectures in a fair benchmark.

Run the complete Kvasir comparison on Epito:

```bash
python scripts/run_benchmark.py --config configs/benchmark_kvasir_epito.yaml
```

BUSI:

```bash
python scripts/run_benchmark.py --config configs/benchmark_busi_epito.yaml
```

See `docs/MODEL_COMPARISONS.md` for the experimental protocol and ablation order.
