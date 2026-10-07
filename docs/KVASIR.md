# Kvasir-SEG: download, prepare, inspect, train, and evaluate

Official dataset: https://datasets.simula.no/kvasir-seg/
Authors' documentation: https://github.com/DebeshJha/Kvasir-SEG

The release contains 1,000 RGB polyp images with corresponding segmentation
masks. Download **kvasir-seg.zip** from the official page's Download section.
The provider restricts use to research and education and requires citation of
Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset," MMM 2020.

## 1. Download and extract

Save the archive from the official page to your Downloads folder. Then run:

```bash
mkdir -p /Users/rashid/data/DS/Healthcare/PB_U-NET/raw/kvasir_seg
unzip ~/Downloads/kvasir-seg.zip -d /Users/rashid/data/DS/Healthcare/PB_U-NET/raw/kvasir_seg
```

Check that the extracted folders are:

```text
/Users/rashid/data/DS/Healthcare/PB_U-NET/raw/kvasir_seg/Kvasir-SEG/
├── images/
└── masks/
```

If the archive extracts with a different enclosing directory, set
`dataset.raw_root` in both `configs/kvasir_pb.yaml` and
`configs/kvasir_unet.yaml` to the folder containing `images/` and `masks/`.
All following commands run from the PB-UNET repository root.

## 2. Validate and prepare

```bash
python scripts/prepare_kvasir.py --config configs/kvasir_pb.yaml --dry-run
python scripts/prepare_kvasir.py --config configs/kvasir_pb.yaml
```

Preparation requires exactly 1,000 unique matching filename stems, readable
files, and equal image/mask dimensions. Images retain their decoded color
intensities at original resolution and are saved as lossless PNGs. Masks are
decoded as grayscale, thresholded at >127, and saved as 0/255 PNGs. Source files
are unchanged; rerunning preparation rewrites the generated files.

Outputs go to the configured `dataset.prepared_root`:

```text
data/kvasir_seg/
├── images/
├── masks/
└── metadata/
    ├── manifest.csv
    ├── preparation.json
    ├── train_split.csv
    ├── val_split.csv
    └── test_split.csv
```

The manifest includes source paths, source SHA-256 checksums, original sizes,
and foreground pixel counts. Preparation saves a deterministic image-level
700/150/150 split with seed 42. Both model configs consume these same CSVs.
This is our experiment split, not an official benchmark split. Patient IDs are
not available in this pipeline; patient independence is not established.

## 3. Inspect samples and model readiness

```bash
python scripts/check_dataset.py --config configs/kvasir_pb.yaml --require-prepared
python scripts/check_dataset.py --config configs/kvasir_unet.yaml --require-prepared
```

Open `runs/smoke_test/kvasir_seg/<timestamp>/sample_00.png` and the other samples.
Each has labeled Image, Binary mask, and Mask overlay panels. The JSON report
and architecture text identify which model was checked.

Training/preview preprocessing resizes RGB images to 256×256 with bilinear
interpolation, scales values to [0,1], and resizes binary masks with nearest
neighbor interpolation. Three color channels are preserved. Augmentation is
disabled in the supplied configs.

## 4. Train both models

```bash
python scripts/train.py --config configs/kvasir_pb.yaml
python scripts/train.py --config configs/kvasir_unet.yaml
```

Defaults are filters [32,64,128], BCEWithLogitsLoss, Adam, learning rate 0.0001,
batch size 8, and up to 100 epochs with early stopping. Outputs, checkpoints,
and training plots go to `runs/kvasir_pb` and `runs/kvasir_unet`.
For a short execution check, add `--epochs 1`.

## 5. Evaluate and compare

```bash
python scripts/evaluate.py --config configs/kvasir_pb.yaml
python scripts/evaluate.py --config configs/kvasir_unet.yaml
python scripts/evaluate.py --config configs/kvasir_pb.yaml --postprocess
python scripts/evaluate.py --config configs/kvasir_unet.yaml --postprocess
python scripts/compare_models.py --config configs/comparison_kvasir.yaml
python scripts/bootstrap_comparison.py --config configs/comparison_kvasir.yaml
```

Evaluation saves CSV/JSON metrics and PNG/PDF plots. Raw and postprocessed
outputs have different suffixes. The comparison config selects raw results;
copy/edit it to compare postprocessed results separately.

## Cluster use

Copy each experiment YAML and change `dataset.raw_root`,
`dataset.prepared_root`, `dataset.images`, `dataset.masks`, the three
`train.*_split` paths, and `evaluate.split_csv` to the cluster dataset location.
Model/run settings remain in config; no dataset path flags are required.
