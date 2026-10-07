"""Model definitions reconstructed from the original 2022 code.

The two public builders intentionally preserve the architecture used by the paper,
while avoiding the original in-place ``list.reverse()`` mutation bug.
"""
from __future__ import annotations

from typing import Sequence
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Concatenate, Conv2D, Dropout,
    Input, MaxPool2D, UpSampling2D,
)


def conv_block(x, num_filters: int):
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def build_unet(
    image_size: int = 256,
    channels: int = 3,
    filters: Sequence[int] = (32, 64, 128),
) -> Model:
    """Baseline U-Net corresponding to original ``model.py``.

    With image_size=256, channels=3 and filters=(32,64,128), Keras reports
    1,213,953 total parameters, matching Table 1 of the manuscript.
    """
    filters = list(filters)
    inputs = Input((image_size, image_size, channels))
    x = inputs
    skips = []
    for f in filters:
        x = conv_block(x, f)
        skips.append(x)
        x = MaxPool2D(2)(x)

    x = conv_block(x, filters[-1])

    for f, skip in zip(reversed(filters), reversed(skips)):
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, skip])
        x = conv_block(x, f)

    x = Conv2D(1, 1, padding="same")(x)
    outputs = Activation("sigmoid")(x)
    return Model(inputs, outputs, name="baseline_unet")


def _portable_bridge(x, reversed_filters, reversed_encoder_skips, n: int = 1):
    """Portable bridge reconstructed from original ``prop_model.portable_bridge``."""
    stage1 = []
    stage2 = []

    # In the original code this convolution occurs before num_filters.reverse().
    x = conv_block(x, reversed_filters[0])

    # PB1: upsample + encoder fusion + convolution.
    for i in range(len(reversed_filters) - n):
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, reversed_encoder_skips[i]])
        x = conv_block(x, reversed_filters[i])
        stage1.append(x)

    stage1 = list(reversed(stage1))

    # PB2: convolution + PB1 fusion + downsample.
    for i in range(len(reversed_filters) - n):
        x = conv_block(x, reversed_filters[i])
        x = Concatenate()([x, stage1[i]])
        stage2.append(x)
        x = MaxPool2D(2)(x)

    return x, list(reversed(stage2))


def build_portable_bridge_unet(
    image_size: int = 256,
    channels: int = 3,
    filters: Sequence[int] = (32, 64, 128),
    encoder_dropout: float = 0.3,
    output_dropout: float = 0.1,
) -> Model:
    """Portable-Bridge U-Net corresponding to original ``prop_model.py``.

    This function preserves the paper architecture but fixes the original mutable
    ``num_filters.reverse()`` side effect. For the default configuration Keras
    reports 2,360,321 parameters, matching Table 1 of the manuscript.
    """
    filters = list(filters)
    inputs = Input((image_size, image_size, channels))
    x = inputs
    encoder_skips = []

    for f in filters:
        x = conv_block(x, f)
        encoder_skips.append(x)
        x = MaxPool2D(2)(x)
        x = Dropout(encoder_dropout)(x)

    reversed_filters = list(reversed(filters))
    reversed_encoder_skips = list(reversed(encoder_skips))
    x, bridge_skips = _portable_bridge(
        x, reversed_filters, reversed_encoder_skips, n=1
    )

    for i in range(len(reversed_filters) - 1):
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, bridge_skips[i]])
        x = conv_block(x, reversed_filters[i])

    x = UpSampling2D(2)(x)
    # Original prop_model uses skip_x[-1] after reversing skip_x, i.e. the
    # shallowest encoder feature map.
    x = Concatenate()([x, reversed_encoder_skips[-1]])
    x = Dropout(output_dropout)(x)
    x = Conv2D(1, 1, padding="same")(x)
    outputs = Activation("sigmoid")(x)
    return Model(inputs, outputs, name="portable_bridge_unet")
