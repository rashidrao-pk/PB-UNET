# Datasets used by the three requested studies

This review covers dataset choice and evaluation setup, rather than attempting
to reproduce the architectures or copy their reported scores into our results.

| Study | Datasets used | Imaging and target | Relevance here |
| --- | --- | --- | --- |
| [LV-UNet, version 2, §V-A](https://arxiv.org/html/2408.16886v2) | ISIC 2016; BUSI; CVC-ClinicDB; CVC-ColonDB; Kvasir-SEG | RGB skin lesions; breast ultrasound lesions; RGB endoscopic polyps | ISIC 2016 is added in this update; BUSI and Kvasir already exist |
| [Bridged-U-Net-ASPP-EVO, §3.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10453237/) | BraTS 2020 and BraTS 2021 | Four-sequence 3D brain MRI; enhancing tumor, tumor core, whole tumor | Multi-modal MRI is not an RGB dataset; it would need a separate volumetric/multi-label pipeline |
| [S-UNet, §III-A](https://doi.org/10.1109/ACCESS.2019.2940476) ([paper PDF](https://scispace.com/pdf/s-unet-a-bridge-style-u-net-framework-with-a-saliency-11ro5843xr.pdf)) | DRIVE; CHASE_DB1; TONGREN clinical dataset | Retinal fundus images and vessel annotations | DRIVE/CHASE are candidates for future RGB retinal work; neither is integrated by this update |

## LV-UNet protocol differences

LV-UNet uses ISIC's 900-image training set and 379-image test set, holding out
20% of training for validation. Its BUSI experiment includes all 780 images,
including normal cases. ClinicDB has 612 pairs, ColonDB 380, and Kvasir 1000.
For these four datasets it uses 20% for test, then an 80/20 training/validation
split of the remainder. It trains for 300 epochs at 256×256 using Adam,
cosine scheduling, and a loss of `0.5 * BCE + Dice`. [Source](https://arxiv.org/html/2408.16886v2)

Our ISIC split sizes follow that holdout structure, with explicit seed 42; exact
paper image identities are not established. Our existing BUSI is lesion-only,
and Kvasir uses a different split. Our standard loss weights, scheduler,
augmentation, training budget, checkpoint rule, and models also differ. Thus,
matching the dataset name does not make a paper score directly comparable.
Use paired results from models trained under one shared repo protocol.

## Brain MRI study

The brain study reports BraTS 2020 with 369 training/125 validation volumes and
BraTS 2021 with 1251 training/219 validation volumes. Its inputs contain T1,
contrast-enhanced T1, T2, and FLAIR sequences. It evaluates three tumor regions
using a 3D model. These are MRI channels, not RGB color channels; converting
illustrative MRI slices to RGB would not reproduce this task. [Source, Table 1 and §3.1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10453237/)

## Retinal study

S-UNet uses DRIVE (40 images; 20 training/20 test), CHASE_DB1 (28 images from
14 children, four-fold evaluation), and the authors' TONGREN clinical set
(30 images; 15 training/15 test). Its vessel targets and field-of-view masks
require retinal-specific evaluation. [Source, §III-A](https://scispace.com/pdf/s-unet-a-bridge-style-u-net-framework-with-a-saliency-11ro5843xr.pdf)

For future CHASE work, keep both eyes from a child in one split rather than
assuming image-level separation implies subject-level separation. For any
retinal integration, specify annotator, field-of-view scoring, resolution, and
patch-vs-image evaluation. Our generic full-image binary metrics cannot be
advertised as an exact retinal-paper protocol. Consult the
[2023 S-UNet correction](https://doi.org/10.1109/ACCESS.2023.3302183) before
reusing numerical claims; correction details were not independently audited here.

## Implemented choice

ISIC 2016 Task 1 adds a new RGB modality and target while fitting our existing
binary segmentation models. It has an official held-out test set and public
image/mask downloads. This is the selected integration, not ISIC 2017/2018 or
the ISIC classification tasks. See [ISIC setup and reproducibility](DATASETS/ISIC2016.md).
