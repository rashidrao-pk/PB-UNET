# Portable-Bridge U-Net — PyTorch edition

This is the **active PyTorch rewrite** of the supplied Portable-Bridge U-Net project. TensorFlow/Keras is no longer required for training, evaluation, inference, metrics, or post-processing.

The rewrite preserves the architecture actually encoded in the paper-era source while fixing the reproducibility problems identified during the audit.

## What was converted

| Old TensorFlow/Keras component | PyTorch replacement |
|---|---|
| `model.py` baseline U-Net | `src/portable_bridge_unet/models.py::BaselineUNet` |
| `prop_model.py` proposed network | `src/portable_bridge_unet/models.py::PortableBridgeUNet` |
| `tf.data` image pipeline | `torch.utils.data.Dataset/DataLoader` |
| Keras BCE | `torch.nn.BCEWithLogitsLoss` |
| Keras Adam | `torch.optim.Adam` |
| `ModelCheckpoint` | PyTorch checkpoint `best.pt` |
| `ReduceLROnPlateau` | `torch.optim.lr_scheduler.ReduceLROnPlateau` |
| Keras training loop | `scripts/train.py` |
| prediction/evaluation notebooks | `scripts/evaluate.py`, `scripts/predict.py` |
| skimage post-processing | OpenCV hole fill + largest connected component |

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
pip install -e .
pip install pytest
```

For an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build for your machine before `pip install -e .` if needed.

## Train the exact paper-era architecture

The implementation defaults to the code's actual 3-channel input, 256x256 images, BCE, Adam and filters 32/64/128:

```bash
python scripts/train.py \
  --images '/path/to/images/**/*.*' \
  --masks '/path/to/masks/**/*.*' \
  --model portable_bridge \
  --channels 3 \
  --image-size 256 \
  --batch-size 8 \
  --lr 1e-4 \
  --epochs 100 \
  --out-dir runs/pb_unet
```

Baseline:

```bash
python scripts/train.py \
  --images '/path/to/images/**/*.*' \
  --masks '/path/to/masks/**/*.*' \
  --model unet \
  --channels 3 \
  --image-size 256 \
  --batch-size 8 \
  --lr 1e-4 \
  --epochs 100 \
  --out-dir runs/unet
```

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
python scripts/evaluate.py \
  --checkpoint runs/pb_unet/best.pt \
  --split-csv runs/pb_unet/test_split.csv \
  --out-dir runs/pb_unet/evaluation
```

With the manuscript post-processing:

```bash
python scripts/evaluate.py \
  --checkpoint runs/pb_unet/best.pt \
  --split-csv runs/pb_unet/test_split.csv \
  --postprocess \
  --out-dir runs/pb_unet/evaluation
```

The evaluator reports **every image**, without the legacy script's score-based filtering or duplicate appending. Outputs include per-image CSV and mean/std/median for Dice, IoU, precision, recall, accuracy and HD95.

## Predict one image

```bash
python scripts/predict.py \
  --checkpoint runs/pb_unet/best.pt \
  --image /path/to/image.png \
  --output prediction.png \
  --postprocess
```

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

Tests verify output dimensions, exact trainable parameter parity, and the post-processing behavior.

## Recommended next step

Do **not** convert the questionable historical H5 weights blindly. Re-run the baseline and proposed models from scratch with this deterministic PyTorch implementation, save the complete test-set metrics, and use those results as the trusted foundation for the revised manuscript and external-dataset experiments.
