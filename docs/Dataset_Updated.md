# Datasets for Portable-Bridge U-Net Experiments

This document describes the datasets planned for the PyTorch reproduction and cross-domain validation of **Portable-Bridge U-Net (PB-U-Net)**.

Recommended order:

1. **Sunnybrook / MICCAI 2009 Cardiac MRI** — reproduce the original paper.
2. **BUSI** — cross-modality validation on breast ultrasound.
3. **Kvasir-SEG** — RGB endoscopic polyp segmentation.
4. **CVC-ClinicDB** — additional polyp training data.
5. **CVC-300, CVC-ColonDB, ETIS-LaribPolypDB** — external cross-dataset testing.
6. **PRS-Med** — optional source/benchmark collection; not required for the first PB-U-Net experiments.

The aim is to keep the model architecture unchanged and adapt only the dataset preparation / loaders.

---

## 1. Recommended local directory layout

Create one dataset root outside the repository:

```text
datasets/
├── sunnybrook/
│   ├── raw/
│   ├── images/
│   ├── masks/
│   └── metadata/
├── busi/
│   ├── raw/
│   ├── images/
│   ├── masks/
│   └── metadata/
├── kvasir_seg/
│   ├── raw/
│   ├── images/
│   └── masks/
├── cvc_clinicdb/
│   ├── raw/
│   ├── images/
│   └── masks/
├── cvc_300/
│   ├── raw/
│   ├── images/
│   └── masks/
├── cvc_colondb/
│   ├── raw/
│   ├── images/
│   └── masks/
├── etis_larib/
│   ├── raw/
│   ├── images/
│   └── masks/
└── prs_med/
    ├── raw/
    └── annotations/
```

Example on macOS:

```bash
mkdir -p ~/datasets/PB_UNET/{sunnybrook,busi,kvasir_seg,cvc_clinicdb,cvc_300,cvc_colondb,etis_larib,prs_med}
```

You may instead use your existing data root, e.g.:

```text
/Users/rashid/data/DS/PB_UNET/
```

---

# 2. Sunnybrook Cardiac Data / MICCAI 2009 LV Segmentation Challenge

## Why we need it

This is the dataset corresponding to the original cardiovascular segmentation work and should be used first to verify the PyTorch implementation.

The Cardiac Atlas Project describes this as the **Sunnybrook Cardiac Data (SCD)**, also known as the **2009 Cardiac MR Left Ventricle Segmentation Challenge data**.

## Official source

https://www.cardiacatlas.org/sunnybrook-cardiac-data/

The page provides:

- DICOM image batch 1
- DICOM image batch 2
- DICOM image batch 3
- DICOM image batch 4
- DICOM image batch 5
- patient metadata CSV
- LV models
- manual contours

## Download

Download all DICOM batches plus:

- `Patient data`
- `Manual contours`

Keep the original files untouched under:

```text
datasets/sunnybrook/raw/
```

Suggested raw layout:

```text
sunnybrook/raw/
├── dicom_batch_1/
├── dicom_batch_2/
├── dicom_batch_3/
├── dicom_batch_4/
├── dicom_batch_5/
├── patient_data.csv
└── contours/
```

## Important reproducibility note

For the **legacy reproduction experiment**, we may first reproduce the old image-level split.

For the **scientifically preferred experiment**, use a **patient-wise split**:

```text
patient A -> train only
patient B -> validation only
patient C -> test only
```

Do not allow slices/frames from one patient to occur in more than one split.

## Expected processed layout

```text
sunnybrook/
├── images/
├── masks/
└── metadata/
    └── samples.csv
```

Recommended `samples.csv`:

```csv
image,mask,patient_id,frame_id,slice_id
images/...,masks/...,SC-HF-I-01,...,...
```

## Experimental role

```text
E0-R  Original-style reproduction
E0-A  U-Net vs PB-U-Net vs PB-U-Net + post-processing
E0-P  Patient-wise split
```

---

# 3. BUSI — Breast Ultrasound Images Dataset

## Why we need it

BUSI gives a strong cross-modality test:

```text
Cardiac MRI -> Breast Ultrasound
```

The original BUSI dataset described by Al-Dhabyani et al. contains **780 breast ultrasound images** from **600 women**, with normal, benign and malignant cases. Ground-truth masks accompany lesion images.

## Primary sources

Dataset information page:

https://scholar.cu.edu.eg/?q=afahmy/pages/dataset

Original publication:

https://doi.org/10.1016/j.dib.2019.104863

Convenient Hugging Face mirror:

https://huggingface.co/datasets/MedOtter/BUSI

## Easy download option: Hugging Face CLI

Install the CLI if necessary:

```bash
pip install -U huggingface_hub
```

Download:

```bash
hf download MedOtter/BUSI \
  --repo-type dataset \
  --local-dir ~/datasets/PB_UNET/busi/raw
```

Older Hugging Face CLI syntax:

```bash
huggingface-cli download MedOtter/BUSI \
  --repo-type dataset \
  --local-dir ~/datasets/PB_UNET/busi/raw
```

## Typical original organization

```text
Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/
```

Lesion masks usually contain `_mask` in the filename.

Some images can have more than one mask, e.g.:

```text
benign (1).png
benign (1)_mask.png
benign (1)_mask_1.png
```

If several masks belong to one image, merge them with pixel-wise OR into one binary mask.

## Samples to use

For the main lesion-segmentation experiment use:

- benign images with masks
- malignant images with masks

Normal images may later be included as all-zero-mask negative examples, but they are not required for the first experiment.

## Expected processed layout

```text
busi/
├── images/
├── masks/
└── metadata/
    └── samples.csv
```

Example manifest:

```csv
image,mask,class
images/benign_0001.png,masks/benign_0001.png,benign
```

## Split recommendation

If patient IDs are recoverable, use patient-wise splits. Otherwise use a fixed-seed image-level split, e.g.:

```text
70% train
15% validation
15% test
```

## Experimental role

```text
E1  BUSI external modality validation
```

---

# 4. Kvasir-SEG

## Why we need it

Kvasir-SEG is an RGB polyp-segmentation dataset and gives a strong domain shift:

```text
MRI grayscale -> RGB gastrointestinal endoscopy
```

## Official source

https://datasets.simula.no/kvasir-seg/

Kvasir-SEG contains **1000 polyp images and corresponding masks**.

## Download

Open the official page and download the dataset archive:

https://datasets.simula.no/kvasir-seg/

Extract under:

```text
datasets/kvasir_seg/raw/
```

## Processed layout

```text
kvasir_seg/
├── images/
├── masks/
└── metadata/
    └── samples.csv
```

The image and mask basename must correspond.

## Channels

Use:

```text
in_channels = 3
```

Do not convert the RGB endoscopy images to grayscale for the main experiment.

## Experimental role

Two valid strategies exist.

### Strategy A — independent Kvasir experiment

```text
Kvasir train
Kvasir validation
Kvasir test
```

### Strategy B — preferred final-paper setup

Train on:

```text
Kvasir-SEG + CVC-ClinicDB
```

and test on unseen:

```text
CVC-300
CVC-ColonDB
ETIS-LaribPolypDB
```

---

# 5. CVC-ClinicDB

## Why we need it

CVC-ClinicDB is commonly paired with Kvasir-SEG for polyp-segmentation training.

## Official source

https://polyp.grand-challenge.org/CVCClinicDB/

## Download

Prefer the official Grand Challenge source.

Place the downloaded archive under:

```text
datasets/cvc_clinicdb/raw/
```

Normalize to:

```text
cvc_clinicdb/
├── images/
├── masks/
└── metadata/
    └── samples.csv
```

## Recommended final-paper training setup

Combine:

```text
Kvasir-SEG + CVC-ClinicDB
```

for train/validation, and keep the datasets below completely unseen during training.

---

# 6. CVC-300

## Why we need it

Use CVC-300 as an **external test set** for cross-dataset polyp generalization.

## Source

https://pages.cvc.uab.es/CVC-Colon/index.php/databases/cvc-endoscenestill/

## Experimental protocol

```text
TRAIN: Kvasir-SEG + CVC-ClinicDB
TEST:  CVC-300
```

Do not train or tune on CVC-300.

## Processed layout

```text
cvc_300/
├── images/
└── masks/
```

---

# 7. CVC-ColonDB

## Why we need it

CVC-ColonDB is another commonly used external polyp-segmentation benchmark.

## Original source

http://mv.cvc.uab.es/projects/colon-qa/cvc-colondb/

If the original server is temporarily unavailable, use a known public mirror but cite the original dataset publication in the paper.

## Download destination

```text
datasets/cvc_colondb/raw/
```

## Processed layout

```text
cvc_colondb/
├── images/
└── masks/
```

## Experimental protocol

```text
TRAIN: Kvasir-SEG + CVC-ClinicDB
TEST:  CVC-ColonDB
```

---

# 8. ETIS-LaribPolypDB

## Why we need it

ETIS is a difficult external test set and is useful for measuring cross-dataset robustness.

## Official source

https://polyp.grand-challenge.org/ETISLarib/

## Download destination

```text
datasets/etis_larib/raw/
```

## Processed layout

```text
etis_larib/
├── images/
└── masks/
```

## Experimental protocol

```text
TRAIN: Kvasir-SEG + CVC-ClinicDB
TEST:  ETIS-LaribPolypDB
```

Do not include ETIS images during hyperparameter tuning.

---

# 9. PRS-Med — optional

## Why it is optional

PRS-Med is useful as:

- a reference benchmark,
- a source of multiple medical-imaging domains,
- a future dataset expansion route.

For the first PB-U-Net paper extension, we do **not** need to reproduce the complete PRS-Med reasoning pipeline.

## Official sources

Project:

https://huyquoctrinh.github.io/prsmed/

GitHub:

https://github.com/huyquoctrinh/PRS-Med

Hugging Face dataset:

https://huggingface.co/datasets/huyquoctrinh/PRS-Med

Paper:

https://arxiv.org/abs/2505.11872

## Download

```bash
hf download huyquoctrinh/PRS-Med \
  --repo-type dataset \
  --local-dir ~/datasets/PB_UNET/prs_med/raw
```

Alternative:

```bash
huggingface-cli download huyquoctrinh/PRS-Med \
  --repo-type dataset \
  --local-dir ~/datasets/PB_UNET/prs_med/raw
```

## Important

Do not treat every PRS-Med QA record as a separate segmentation sample.

PRS-Med expands source images into multiple reasoning / question-answer examples, so for PB-U-Net segmentation experiments we must first identify unique:

```text
image -> mask
```

pairs.

---

# 10. Recommended experiment matrix

| Experiment | Train | Validation | Test | Purpose |
|---|---|---|---|---|
| E0-R | Sunnybrook | Sunnybrook | Sunnybrook | reproduce original paper |
| E0-A | Sunnybrook | Sunnybrook | Sunnybrook | U-Net / PB / PB+PP ablation |
| E0-P | Sunnybrook patients | Sunnybrook patients | Sunnybrook patients | patient-wise validation |
| E1 | BUSI | BUSI | BUSI | ultrasound transfer |
| E2 | Kvasir + ClinicDB | held-out train data | Kvasir/ClinicDB | RGB polyp segmentation |
| E3-A | Kvasir + ClinicDB | held-out train data | CVC-300 | external generalization |
| E3-B | Kvasir + ClinicDB | held-out train data | ColonDB | external generalization |
| E3-C | Kvasir + ClinicDB | held-out train data | ETIS | external generalization |

Do **not** tune model hyperparameters using E3-A/B/C.

---

# 11. Models to run on every dataset

At minimum:

```text
1. Baseline U-Net
2. Portable-Bridge U-Net
3. Portable-Bridge U-Net + post-processing
```

Use the same architecture/hyperparameter policy unless a change is explicitly reported as a separate ablation.

---

# 12. Metrics

Calculate metrics over **every image in the test set**:

```text
Dice
IoU
Precision
Recall
HD95
```

Report:

```text
mean ± standard deviation
median
```

Optionally also report:

```text
parameters
FLOPs
inference time / image
```

Never remove low-performing samples from aggregate metrics.

---

# 13. Image preprocessing

## Cardiac MRI

For exact legacy reproduction:

```text
256 x 256
3 channels
```

This reproduces the historical implementation, which loaded grayscale MRI into a 3-channel representation.

For the clean follow-up experiment also test:

```text
256 x 256
1 channel
```

and report it separately.

## BUSI

Recommended main setup:

```text
256 x 256
1 channel
```

A 3-channel replicated-grayscale version can be used as an ablation if exact architectural consistency across domains is desired.

## Polyp datasets

Use:

```text
256 x 256
3 channels RGB
```

Masks:

```text
256 x 256
1 binary channel
```

Always resize masks with **nearest-neighbor interpolation**.

---

# 14. Mask sanity checks

Before training, verify:

```python
print(image.shape)
print(mask.shape)
print(image.min(), image.max())
print(np.unique(mask))
```

Expected after preprocessing:

```text
image: C x H x W
mask:  1 x H x W

image values: approximately [0, 1]
mask values:  [0, 1]
```

Never resize segmentation masks with bilinear interpolation.

---

# 15. Recommended preparation scripts

The repository should eventually contain:

```text
scripts/
├── prepare_sunnybrook.py
├── prepare_busi.py
├── prepare_kvasir.py
├── prepare_cvc_clinicdb.py
├── prepare_cvc300.py
├── prepare_colondb.py
└── prepare_etis.py
```

Each script should create a common manifest.

---

# 16. Common manifest format

Recommended schema:

```csv
sample_id,image,mask,dataset,patient_id,split
```

Example:

```csv
busi_0001,/data/busi/images/0001.png,/data/busi/masks/0001.png,BUSI,,train
sunny_0001,/data/sunny/images/0001.png,/data/sunny/masks/0001.png,Sunnybrook,SC-HF-I-01,test
```

This allows all experiments to use one common PyTorch dataset class.

---

# 17. Verify downloads before training

For every dataset, record:

```text
number of images
number of masks
number of valid image-mask pairs
number of missing masks
number of duplicate filenames
number of empty masks
```

Example summary:

```text
DATASET            IMAGES   MASKS   VALID PAIRS
Sunnybrook         ...      ...     ...
BUSI               ...      ...     ...
Kvasir-SEG         1000     1000    1000
CVC-ClinicDB       ...      ...     ...
CVC-300            ...      ...     ...
CVC-ColonDB        ...      ...     ...
ETIS                ...      ...     ...
```

Do not start full training until these counts are verified.

---

# 18. Suggested download order

To avoid unnecessary work, proceed in this order:

```text
1. Sunnybrook
2. BUSI
3. Kvasir-SEG
4. CVC-ClinicDB
5. CVC-300
6. CVC-ColonDB
7. ETIS-LaribPolypDB
8. PRS-Med only if needed
```

---

# 19. Citation reminder

When a dataset is used in the paper, cite the **original dataset/publication**, not only a mirror or hosting page.

For BUSI, cite:

> Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. *Dataset of breast ultrasound images.* Data in Brief, 28, 104863.

For Kvasir-SEG, Sunnybrook, CVC datasets and ETIS, use the citation requested on the corresponding official dataset page.

---

# 20. Final recommended paper setup

```text
                    Portable-Bridge U-Net
                           |
          +----------------+----------------+
          |                |                |
     Cardiac MRI      Breast US       RGB Endoscopy
     Sunnybrook          BUSI       Kvasir + ClinicDB
                                            |
                             +--------------+--------------+
                             |              |              |
                           CVC-300        ColonDB          ETIS
                           unseen          unseen          unseen
```

This provides:

- original-task reproduction,
- modality transfer,
- anatomy transfer,
- RGB vs grayscale validation,
- external cross-dataset generalization.


---

# 21. Pre-Training Smoke Test Checklist

Before launching any full experiment on Sunnybrook, BUSI, Kvasir-SEG, or the external polyp datasets, run a small smoke test first.

The purpose is to verify that the complete PyTorch pipeline works correctly from data loading through prediction.

## What we want to verify next

1. **The dataloader reads image/mask pairs correctly.**
2. **The mask is binary and spatially aligned with the corresponding image.**
3. **The training loss decreases and does not produce NaNs or infinities.**
4. **A checkpoint is saved successfully.**
5. **The evaluation pipeline runs successfully on the saved checkpoint.**
6. **Prediction masks have the expected shape and value range.**

## A. Verify one dataloader batch

Before training, inspect one batch.

Expected tensor shapes:

```text
image: (B, C, 256, 256)
mask:  (B, 1, 256, 256)
```

For the original cardiac reproduction:

```text
C = 3
```

For native grayscale experiments such as BUSI or single-channel cardiac MRI:

```text
C = 1
```

For RGB datasets such as Kvasir-SEG:

```text
C = 3
```

Check the value ranges:

```python
print("image shape:", images.shape)
print("mask shape :", masks.shape)

print("image min/max:", images.min().item(), images.max().item())
print("mask unique:", masks.unique())
```

Expected:

```text
image values: approximately 0.0 to 1.0
mask values:  0.0 and 1.0
```

If the mask contains intermediate values, it has probably been resized incorrectly. Segmentation masks must be resized using **nearest-neighbor interpolation** and then binarized.

---

## B. Visually verify image-mask alignment

Do not rely only on tensor dimensions.

For several random samples, overlay the binary mask on the input image and verify that the mask corresponds to the correct anatomical structure or lesion.

Check for:

- incorrect image-mask pairing,
- horizontal or vertical flips affecting only one of the pair,
- resize mismatch,
- mask shifted relative to the image,
- incorrect orientation,
- corrupted masks,
- empty masks when a lesion should be present.

A simple visual sanity check should be performed before every new dataset experiment.

---

## C. Run a one-epoch smoke training

Before a 50- or 100-epoch experiment, run only one epoch with a small batch size.

Example:

```bash
python scripts/train.py \
  --images '/path/to/images/*.png' \
  --masks '/path/to/masks/*.png' \
  --model portable_bridge \
  --channels 3 \
  --image-size 256 \
  --batch-size 2 \
  --lr 1e-4 \
  --epochs 1 \
  --device auto \
  --out-dir runs/smoke_pb
```

Also smoke-test the baseline:

```bash
python scripts/train.py \
  --images '/path/to/images/*.png' \
  --masks '/path/to/masks/*.png' \
  --model unet \
  --channels 3 \
  --image-size 256 \
  --batch-size 2 \
  --lr 1e-4 \
  --epochs 1 \
  --device auto \
  --out-dir runs/smoke_unet
```

The goal is not to obtain meaningful segmentation performance after one epoch. We only need to verify that:

```text
forward pass works
loss is finite
backpropagation works
optimizer.step() works
validation completes
checkpoint writing works
```

---

## D. Verify the loss

During the smoke run, check that the loss is finite.

Valid:

```text
train_loss = 0.71
val_loss   = 0.66
```

Invalid:

```text
train_loss = nan
train_loss = inf
```

The loss does not need to fall dramatically during one epoch, but over several batches it should normally remain stable or begin to decrease.

If NaNs occur, inspect:

1. input normalization,
2. mask values,
3. learning rate,
4. empty/corrupted samples,
5. numerical operations in the model.

---

## E. Verify checkpoint creation

After training, confirm that the run directory contains the expected experiment artifacts.

For example:

```text
runs/smoke_pb/
├── best.pt
├── config.json
├── history.json
├── train_split.csv
├── val_split.csv
└── test_split.csv
```

The exact filenames may evolve, but at minimum the following must be retained:

```text
model weights
training configuration
data split
training history
```

The dataset split is especially important for reproducibility.

---

## F. Verify evaluation

Run evaluation using the saved checkpoint rather than the in-memory model.

Example:

```bash
python scripts/evaluate.py \
  --checkpoint runs/smoke_pb/best.pt \
  --split-csv runs/smoke_pb/test_split.csv \
  --out-dir runs/smoke_pb/evaluation
```

The evaluation must complete over **all test samples**.

Expected metrics:

```text
Dice
IoU
Precision
Recall
HD95
```

For the full experiments report:

```text
mean
standard deviation
median
```

Do not exclude low-performing images.

---

## G. Verify prediction output

Test the prediction script on a few images.

The raw network output should have shape:

```text
(B, 1, H, W)
```

For 256 × 256 inputs:

```text
(B, 1, 256, 256)
```

After sigmoid:

```text
probability map range: 0.0 to 1.0
```

After thresholding:

```text
binary prediction values: {0, 1}
```

The saved mask should preserve the spatial dimensions of the evaluation image or be correctly mapped back to the original image size if that functionality is enabled.

---

## H. Smoke-test pass criteria

A dataset is ready for full training only when all of the following pass:

```text
[ ] Correct number of image-mask pairs
[ ] One dataloader batch loads without error
[ ] Image tensor shape is correct
[ ] Mask tensor shape is correct
[ ] Masks are binary
[ ] Image and mask are visually aligned
[ ] Forward pass works
[ ] Loss is finite
[ ] Backpropagation works
[ ] One epoch completes
[ ] Checkpoint is written
[ ] Evaluation reloads the checkpoint
[ ] Evaluation completes on all samples
[ ] Prediction tensor shape is correct
[ ] Saved prediction visually matches the input
```

Only after this checklist passes should a full experiment such as E0-R, E1, E2, or E3 be launched.
