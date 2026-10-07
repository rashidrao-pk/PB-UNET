# Training

## Create figures for all datasets on Epito

From the repository root in your PyTorch environment:

```bash
python scripts/preview_datasets.py --num-samples 6
```

This uses `paper_legacy_epito.yaml`, `busi_pb_epito.yaml`, and
`kvasir_pb_epito.yaml`. It checks dataset availability and creates the same
bordered, titled **Image / Binary mask / Mask overlay** figures as the existing
smoke tests. Each PNG shows the prepared target mask, not a model prediction.
Resize and normalization match the configured loader, with augmentation disabled.

Outputs are preserved in timestamped directories:

```text
runs/smoke_test/sunnybrook/<timestamp>/sample_00.png
runs/smoke_test/busi/<timestamp>/sample_00.png
runs/smoke_test/kvasir_seg/<timestamp>/sample_00.png
```

Each directory also contains the dataset/model report, architecture text, and
resolved config. A missing dataset produces a failure report; the command still
checks the remaining datasets and exits unsuccessfully if any check fails.
Prepare missing data using the existing dataset preparation guides first.

For one dataset or more samples:

```bash
python scripts/preview_datasets.py --configs configs/kvasir_pb_epito.yaml --num-samples 12
```

The figures use deterministic samples spaced across sorted prepared pairs.
They do not necessarily select the same source image as an older sample number.

Run commands from the repository root. All dataset paths and experiment settings
come from YAML; command-line options override individual settings.

Check the downloaded dataset and prepared training pairs:

```bash
python scripts/check_dataset.py --config configs/paper_legacy.yaml
python scripts/prepare_sunnybrook.py --config configs/paper_legacy.yaml
python scripts/check_dataset.py --config configs/paper_legacy.yaml --require-prepared
```

The configured raw directory is
`/Users/rashid/data/DS/Healthcare/PB_U-NET/raw`.
Raw DICOM images and manual contours must be converted into prepared image/mask
pairs before training. Configure their locations with `dataset.images` and
`dataset.masks`; the defaults use `data/sunnybrook/images/**/*.png` and
`data/sunnybrook/masks/**/*.png` under the repository.

Train PB-U-Net:

```bash
python scripts/train.py --config configs/paper_legacy.yaml
```

Train the baseline:

```bash
python scripts/train.py --config configs/baseline.yaml
```

The configs specify 3 channels, 256×256 input, batch size 8, learning rate
0.0001, and 100 epochs. Run outputs go to `runs/pb_unet` and `runs/unet`
respectively. Keep the dataset settings identical in both configs for comparison.

For a quick experiment, override only the settings you need:

```bash
python scripts/train.py --config configs/paper_legacy.yaml --epochs 1 --device cpu
```

With no `--config`, the trainer uses `configs/paper_legacy.yaml`.
Relative YAML paths resolve against the config file's directory; relative CLI
paths resolve against the current working directory.

Each run saves `best.pt`, `history.json`, resolved arguments in `config.json`,
and the actual train/validation/test split CSV files.

## Review the smoke test before training

```bash
python scripts/check_dataset.py --config configs/paper_legacy.yaml --require-prepared
```

Every dataset check writes a new timestamped folder under `runs/smoke_test`
(configurable through `smoke_test.out_dir`). Open the generated folder and review:

- `sample_*.png`: image, binary mask, and red mask overlay from left to right.
- `report.json`: dataset counts, sample shapes/ranges, preprocessing settings,
  parameter count, model forward-pass result, and any failures.
- `model_architecture.txt`: the configured architecture.
- `resolved_config.json`: resolved experiment settings and dataset paths.
- `README.md`: interpretation and limitations of the check.

The previews use the training resize/normalization code with augmentation
disabled. The model check uses the configured architecture and image size on
CPU with random weights. It does not train a model or assess prediction quality.
The report records the configured training augmentation and split policy.
`--require-prepared` exits unsuccessfully if data or model checks fail.
Previous reports are preserved. Missing data still produces a failure report.

Use `configs/baseline.yaml` to check the baseline architecture. Override
`--num-samples 12` or `--out-dir runs/my_smoke_test` when needed.

## BUSI sample previews

For the prepared BUSI dataset on the cluster:

```bash
python scripts/check_dataset.py --config configs/busi_pb_epito.yaml --require-prepared
```

This saves the same labeled image/mask/overlay PNGs and model report under
`runs/smoke_test/busi/<timestamp>/`. The checker uses BUSI image/mask pairs
without requiring Sunnybrook DICOM directories. For the baseline architecture,
select `configs/BUSI_epito_baseline.yaml`.

These configs reference cluster paths under `/beegfs`. For local use, copy the
config and set `dataset.images` and `dataset.masks` to the local prepared BUSI
files. Previews reflect the configured PNG masks; they do not establish how
the source BUSI annotations were prepared.

## Saved training plots

The trainer updates `training_curves.png` and `training_curves.pdf` in the run
directory after every completed epoch, including an epoch that triggers early
stopping. Curves show training and validation loss, Dice, IoU, precision,
recall, accuracy, and learning rate. Numeric values remain in `history.json`.

To create plots for previously completed runs without retraining:

```bash
python scripts/plot_results.py
```

This scans `runs/` for saved training histories and evaluation CSVs. Use
`--runs-dir runs/pb_unet` to restrict it to one run.
