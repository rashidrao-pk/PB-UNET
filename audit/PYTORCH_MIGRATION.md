# TensorFlow -> PyTorch migration audit

## Confirmed semantic mappings

- Keras `Conv2D(..., padding='same')` -> PyTorch `Conv2d(..., padding=1)` for 3x3 stride 1.
- Keras `BatchNormalization` -> PyTorch `BatchNorm2d`.
- Keras `Activation('relu')` -> PyTorch `ReLU`.
- Keras `MaxPool2D((2,2))` -> PyTorch `MaxPool2d(2,2)`.
- Keras `UpSampling2D((2,2))` default nearest interpolation -> `F.interpolate(..., scale_factor=2, mode='nearest')`.
- Keras `Concatenate()` NHWC -> `torch.cat(..., dim=1)` NCHW.
- Keras `Dropout` -> `Dropout2d` in the segmentation rewrite. Note: this applies channel-wise dropout rather than element-wise dropout; use `nn.Dropout` if exact element-wise Keras semantics are required. For strict parity, see note below.
- Keras final sigmoid + BCE -> logits + `BCEWithLogitsLoss`; mathematically equivalent objective, numerically more stable.

## Strict-parity note on dropout

Keras `Dropout` is element-wise. PyTorch `Dropout2d` is channel-wise. For exact historical semantics the model should use `nn.Dropout`, not `nn.Dropout2d`. The active implementation has been set to `nn.Dropout` for strict parity.

## Parameter parity

Exact trainable count parity was reproduced for both models.

## Historical H5 warning

Inspection of the supplied `model_Proposed_Model_100_0.001.h5` shows the first convolution kernel has shape `(3,3,3,128)`, indicating that checkpoint was built with encoder order `128 -> 64 -> 32`, likely because the mutable `num_filters.reverse()` side effect had already happened. This is not the canonical `32 -> 64 -> 128` configuration described in the manuscript parameter table and cleaned implementation.

Therefore H5 checkpoint migration is intentionally not automated. Retraining is safer than assigning weights to a possibly different architecture.
