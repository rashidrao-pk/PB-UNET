from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers, matching the Keras implementation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineUNet(nn.Module):
    """PyTorch conversion of the original ``model.py`` U-Net.

    Defaults correspond to the manuscript implementation: RGB-shaped input and
    filters (32, 64, 128). The PyTorch trainable parameter count is 1,211,649,
    equal to the Keras manuscript's *trainable* count. Keras also reports 2,304
    non-trainable BatchNorm moving-statistic values, yielding 1,213,953 total.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        filters: Sequence[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        if not filters:
            raise ValueError("filters must not be empty")
        self.filters = tuple(int(f) for f in filters)

        encoders = []
        ch = in_channels
        for f in self.filters:
            encoders.append(ConvBlock(ch, f))
            ch = f
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bridge = ConvBlock(self.filters[-1], self.filters[-1])

        decoders = []
        ch = self.filters[-1]
        for f in reversed(self.filters):
            # nearest-neighbour UpSampling2D is parameter free; interpolation is
            # performed in forward. Concatenation adds f skip channels.
            decoders.append(ConvBlock(ch + f, f))
            ch = f
        self.decoders = nn.ModuleList(decoders)
        self.head = nn.Conv2d(ch, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bridge(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # Return logits. Use BCEWithLogitsLoss during training and sigmoid only
        # for metrics/inference; this is numerically safer than embedding sigmoid.
        return self.head(x)


class PortableBridge(nn.Module):
    """Exact structural conversion of ``prop_model.portable_bridge``.

    For the manuscript's three-level network, this module performs:
      1. bottleneck ConvBlock at 128 channels;
      2. two upsample/encoder-fusion blocks (PB1);
      3. two ConvBlock/PB1-fusion/pool blocks (PB2).

    The original mutable-list reversal bug is deliberately not reproduced.
    """

    def __init__(self, filters: Sequence[int]) -> None:
        super().__init__()
        filters = tuple(int(f) for f in filters)
        if len(filters) < 2:
            raise ValueError("PortableBridge requires at least two filter levels")
        self.filters = filters
        rev = tuple(reversed(filters))
        self.rev_filters = rev

        self.bottleneck = ConvBlock(filters[-1], filters[-1])

        # PB1 channel bookkeeping. Start with bottleneck output filters[-1].
        pb1 = []
        ch = filters[-1]
        # Reversed encoder skips have channels rev[i].
        for i in range(len(filters) - 1):
            f = rev[i]
            skip_ch = rev[i]
            pb1.append(ConvBlock(ch + skip_ch, f))
            ch = f
        self.pb1 = nn.ModuleList(pb1)

        # PB1 features are reversed before PB2 use.
        # Their channels before reversal are rev[:L-1].
        pb1_rev_channels = list(reversed(rev[: len(filters) - 1]))

        pb2 = []
        pb2_fuse_channels = []
        for i in range(len(filters) - 1):
            f = rev[i]
            pb2.append(ConvBlock(ch, f))
            ch_after_concat = f + pb1_rev_channels[i]
            pb2_fuse_channels.append(ch_after_concat)
            ch = ch_after_concat  # pooling does not alter channels
        self.pb2 = nn.ModuleList(pb2)
        self.pb2_fuse_channels = tuple(pb2_fuse_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels = ch

    def forward(
        self, x: torch.Tensor, encoder_skips: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(encoder_skips) != len(self.filters):
            raise ValueError(
                f"expected {len(self.filters)} encoder skips, got {len(encoder_skips)}"
            )
        rev_skips = list(reversed(encoder_skips))
        x = self.bottleneck(x)

        stage1: list[torch.Tensor] = []
        for i, block in enumerate(self.pb1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, rev_skips[i]], dim=1)
            x = block(x)
            stage1.append(x)

        stage1.reverse()

        stage2: list[torch.Tensor] = []
        for i, block in enumerate(self.pb2):
            x = block(x)
            x = torch.cat([x, stage1[i]], dim=1)
            stage2.append(x)
            x = self.pool(x)

        stage2.reverse()
        return x, stage2


class PortableBridgeUNet(nn.Module):
    """PyTorch conversion of the proposed Portable-Bridge U-Net.

    Defaults reproduce the active TensorFlow architecture while fixing the
    original in-place ``list.reverse()`` side effect. The PyTorch trainable
    parameter count is 2,356,609, equal to the Keras manuscript's trainable
    count. Keras additionally counted 3,712 BN moving statistics as
    non-trainable values, for 2,360,321 total.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        filters: Sequence[int] = (32, 64, 128),
        encoder_dropout: float = 0.3,
        output_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(filters) < 2:
            raise ValueError("PortableBridgeUNet requires at least two filter levels")
        self.filters = tuple(int(f) for f in filters)

        encoders = []
        ch = in_channels
        for f in self.filters:
            encoders.append(ConvBlock(ch, f))
            ch = f
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_dropout = nn.Dropout(p=encoder_dropout)

        self.portable_bridge = PortableBridge(self.filters)

        rev = tuple(reversed(self.filters))
        # stage2 is returned reversed. For the 3-level architecture its channel
        # dimensions are [192, 192]. Generalize from bridge bookkeeping.
        bridge_skip_channels = list(reversed(self.portable_bridge.pb2_fuse_channels))

        decoders = []
        ch = self.portable_bridge.out_channels
        for i in range(len(self.filters) - 1):
            f = rev[i]
            decoders.append(ConvBlock(ch + bridge_skip_channels[i], f))
            ch = f
        self.decoders = nn.ModuleList(decoders)

        # Final original decoder step concatenates shallowest encoder feature and
        # directly applies dropout + 1x1 output convolution (no ConvBlock).
        shallow_ch = self.filters[0]
        self.output_dropout = nn.Dropout(p=output_dropout)
        self.head = nn.Conv2d(ch + shallow_ch, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            encoder_skips.append(x)
            x = self.pool(x)
            x = self.encoder_dropout(x)

        x, bridge_skips = self.portable_bridge(x, encoder_skips)

        for i, dec in enumerate(self.decoders):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat([x, bridge_skips[i]], dim=1)
            x = dec(x)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, encoder_skips[0]], dim=1)
        x = self.output_dropout(x)
        return self.head(x)


def build_model(
    name: str,
    in_channels: int = 3,
    out_channels: int = 1,
    filters: Sequence[int] = (32, 64, 128),
) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"unet", "baseline", "baseline_unet"}:
        return BaselineUNet(in_channels, out_channels, filters)
    if key in {"portable_bridge", "portable_bridge_unet", "pb_unet"}:
        return PortableBridgeUNet(in_channels, out_channels, filters)
    raise ValueError(f"unknown model: {name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
