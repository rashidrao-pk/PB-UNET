# Model card

## Scope

This repository implements trainable **binary 2D segmentation architectures**, not
a single validated clinical model. A trained checkpoint's behavior depends on
its dataset, split, preprocessing and training run. No universally applicable
checkpoint or clinical operating point is supplied. Intended use is research,
method comparison and reproducibility studies.

## Implementation identity

| Name | Implementation and claim boundary |
|---|---|
| `unet` | Local baseline with specified widths and BatchNorm; not every published U-Net variant |
| `portable_bridge_legacy` | Paper-era topology and parameter-count match to retained Keras sources; numerical training equivalence unverified |
| `portable_bridge` | Revised bridge model with bilinear alignment, spatial encoder dropout and final refinement |
| `apb_unet` | Revised PB with channel-gated bridge fusion; novelty and benefit require ablations |
| `unetpp` | Local nested dense-skip adaptation, shared widths; no original-paper deep-supervision protocol claim |
| `attention_unet` | Local attention-gated decoder adaptation |
| `resunet` | Local residual encoder/decoder adaptation |
| `resunetpp` | Local residual/SE/ASPP/attention combination; not an audited exact reproduction |
| `unet3plus` | Local full-scale fusion implementation inspired by UNet 3+ |

With three input channels and filters [32,64,128], the baseline has 1,211,649
trainable parameters and legacy PB has 2,356,609. Parameter equality establishes
a useful structural check, not equal initialization, BatchNorm semantics,
optimizer behavior, preprocessing, predictions or convergence across frameworks.
PyTorch layers use their current configured/default initialization and BatchNorm
settings. Do not call this bitwise or numerically exact TensorFlow reproduction.
Historical unversioned checkpoints require compatibility inference and may warn
about ambiguous interpolation; retrain for definitive provenance.

## Inputs and outputs

Inputs are one-channel grayscale or three-channel RGB images resized to the
configured resolution and scaled to [0,1]. Model outputs are one-channel logits;
sigmoid and a declared threshold produce binary masks. BraTS configs select one
MRI modality and one binary region at a time. Four-modality fusion, simultaneous
multiclass BraTS prediction, volume reconstruction and 3D physical-distance
evaluation are not implemented by this workflow.

## Evaluation requirements

Use held-out independent cases/patients when identifiers permit. Assess complete
per-image results, case-level variability, subgroup/failure behavior, external
data, calibration/threshold sensitivity where relevant, and training-seed
variability. No aggregate metric in this repository alone establishes safety,
fairness, generalization, or superiority over another method. See
[reproducibility](REPRODUCIBILITY.md) and [readiness audit](docs/PUBLICATION_READINESS.md).

## Limitations

Resizing can erase small structures; uint8 MRI export discards intensity precision.
Image-level splits can leak correlated observations. Class imbalance makes
accuracy a weak primary score. Empty-mask and HD95 conventions affect rankings.
Largest-component filtering can remove valid anatomy. Dataset shifts, artifacts,
annotation disagreement, demographic representation and acquisition effects
require dedicated evaluation. Data licenses and access terms apply separately
from the MIT software license. Clinical deployment requires separate validation
and appropriate governance, which this research repository does not provide.

## Architecture references

The following primary papers identify comparison families; the table above
states the narrower implementation claims in this repository:

- [UNet++](https://arxiv.org/abs/1807.10165)
- [Attention U-Net](https://arxiv.org/abs/1804.03999)
- [ResUNet++](https://arxiv.org/abs/1911.07067)
- [UNet 3+](https://arxiv.org/abs/2004.08790)

Use the relevant original citations in a manuscript and disclose implementation
and protocol differences rather than presenting these names as exact reproductions.
