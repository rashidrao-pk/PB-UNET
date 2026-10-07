from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(x: torch.Tensor, size=None, scale_factor=2, mode: str = "bilinear") -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(x, size=size, scale_factor=None if size is not None else scale_factor, mode="nearest")
    return F.interpolate(
        x,
        size=size,
        scale_factor=None if size is not None else scale_factor,
        mode="bilinear",
        align_corners=False,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.proj = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class SkipGate(nn.Module):
    """Lightweight channel gate used by APB-U-Net."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.theta = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1)
        self.bn = nn.BatchNorm2d(inter_channels)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        g = upsample(gate, size=skip.shape[-2:]) if gate.shape[-2:] != skip.shape[-2:] else gate
        a = F.relu(self.bn(self.theta(skip) + self.phi(g)), inplace=True)
        a = torch.sigmoid(self.psi(a))
        return skip * a


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch = max(out_channels // 4, 8)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, branch, 1, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_channels, branch, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_channels, branch, 3, padding=4, dilation=4, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(in_channels, branch, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(branch), nn.ReLU(inplace=True)),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(branch * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class BaselineUNet(nn.Module):
    """Paper-era baseline U-Net. Parameter count remains 1,211,649 for [32,64,128]."""

    def __init__(self, in_channels=3, out_channels=1, filters: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        self.filters = tuple(map(int, filters))
        encoders, ch = [], in_channels
        for f in self.filters:
            encoders.append(ConvBlock(ch, f)); ch = f
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(self.filters[-1], self.filters[-1])
        decoders, ch = [], self.filters[-1]
        for f in reversed(self.filters):
            decoders.append(ConvBlock(ch + f, f)); ch = f
        self.decoders = nn.ModuleList(decoders)
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bridge(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = upsample(x, mode="nearest")
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


class PortableBridge(nn.Module):
    def __init__(self, filters: Sequence[int], up_mode: str = "nearest", gated: bool = False) -> None:
        super().__init__()
        filters = tuple(map(int, filters))
        if len(filters) < 2:
            raise ValueError("PortableBridge requires at least two filter levels")
        self.filters = filters
        self.up_mode = up_mode
        rev = tuple(reversed(filters))
        self.bottleneck = ConvBlock(filters[-1], filters[-1])
        pb1, ch = [], filters[-1]
        gates = []
        for i in range(len(filters) - 1):
            f = rev[i]
            pb1.append(ConvBlock(ch + rev[i], f)); ch = f
            gates.append(SkipGate(rev[i]) if gated else nn.Identity())
        self.pb1 = nn.ModuleList(pb1)
        self.gates = nn.ModuleList(gates)
        pb1_rev_channels = list(reversed(rev[: len(filters) - 1]))
        pb2, fused = [], []
        for i in range(len(filters) - 1):
            f = rev[i]
            pb2.append(ConvBlock(ch, f))
            ch = f + pb1_rev_channels[i]
            fused.append(ch)
        self.pb2 = nn.ModuleList(pb2)
        self.pb2_fuse_channels = tuple(fused)
        self.pool = nn.MaxPool2d(2)
        self.out_channels = ch

    def forward(self, x, encoder_skips):
        rev_skips = list(reversed(encoder_skips))
        x = self.bottleneck(x)
        stage1 = []
        for i, block in enumerate(self.pb1):
            x = upsample(x, mode=self.up_mode)
            skip = self.gates[i](rev_skips[i])
            x = block(torch.cat([x, skip], dim=1))
            stage1.append(x)
        stage1.reverse()
        stage2 = []
        for i, block in enumerate(self.pb2):
            x = block(x)
            x = torch.cat([x, stage1[i]], dim=1)
            stage2.append(x)
            x = self.pool(x)
        stage2.reverse()
        return x, stage2


class PortableBridgeLegacyUNet(nn.Module):
    """Exact structural paper-era PB-U-Net for reproducibility."""

    def __init__(self, in_channels=3, out_channels=1, filters: Sequence[int] = (32, 64, 128), encoder_dropout=0.3, output_dropout=0.1) -> None:
        super().__init__()
        self.filters = tuple(map(int, filters))
        encoders, ch = [], in_channels
        for f in self.filters:
            encoders.append(ConvBlock(ch, f)); ch = f
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(2)
        self.encoder_dropout = nn.Dropout(encoder_dropout)
        self.portable_bridge = PortableBridge(self.filters, up_mode="nearest", gated=False)
        rev = tuple(reversed(self.filters))
        bridge_skip_channels = list(reversed(self.portable_bridge.pb2_fuse_channels))
        decoders, ch = [], self.portable_bridge.out_channels
        for i in range(len(self.filters) - 1):
            f = rev[i]
            decoders.append(ConvBlock(ch + bridge_skip_channels[i], f)); ch = f
        self.decoders = nn.ModuleList(decoders)
        self.output_dropout = nn.Dropout(output_dropout)
        self.head = nn.Conv2d(ch + self.filters[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x); skips.append(x); x = self.encoder_dropout(self.pool(x))
        x, bridge_skips = self.portable_bridge(x, skips)
        for i, dec in enumerate(self.decoders):
            x = upsample(x, mode="nearest")
            x = dec(torch.cat([x, bridge_skips[i]], dim=1))
        x = upsample(x, mode="nearest")
        x = self.output_dropout(torch.cat([x, skips[0]], dim=1))
        return self.head(x)


class PortableBridgeUNet(nn.Module):
    """PB-U-Net-R: bilinear bridge/decoder alignment plus a final refinement block."""

    def __init__(self, in_channels=3, out_channels=1, filters: Sequence[int] = (32, 64, 128), encoder_dropout=0.3, output_dropout=0.1, gated=False, decoder_mode="bilinear", spatial_encoder_dropout=True) -> None:
        super().__init__()
        self.filters = tuple(map(int, filters))
        encoders, ch = [], in_channels
        for f in self.filters:
            encoders.append(ConvBlock(ch, f)); ch = f
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(2)
        self.decoder_mode = decoder_mode
        self.encoder_dropout = (nn.Dropout2d if spatial_encoder_dropout else nn.Dropout)(encoder_dropout)
        self.portable_bridge = PortableBridge(self.filters, up_mode="bilinear", gated=gated)
        rev = tuple(reversed(self.filters))
        bridge_skip_channels = list(reversed(self.portable_bridge.pb2_fuse_channels))
        decoders, ch = [], self.portable_bridge.out_channels
        for i in range(len(self.filters) - 1):
            f = rev[i]
            decoders.append(ConvBlock(ch + bridge_skip_channels[i], f)); ch = f
        self.decoders = nn.ModuleList(decoders)
        self.final_refine = ConvBlock(ch + self.filters[0], self.filters[0])
        self.output_dropout = nn.Dropout2d(output_dropout)
        self.head = nn.Conv2d(self.filters[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x); skips.append(x); x = self.encoder_dropout(self.pool(x))
        x, bridge_skips = self.portable_bridge(x, skips)
        for i, dec in enumerate(self.decoders):
            x = upsample(x, mode=self.decoder_mode)
            x = dec(torch.cat([x, bridge_skips[i]], dim=1))
        x = upsample(x, mode="bilinear")
        x = self.final_refine(torch.cat([x, skips[0]], dim=1))
        return self.head(self.output_dropout(x))


class AdaptivePortableBridgeUNet(PortableBridgeUNet):
    """APB-U-Net: PB-U-Net-R with lightweight channel-gated encoder fusion."""
    def __init__(self, in_channels=3, out_channels=1, filters: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__(in_channels, out_channels, filters, gated=True)


class UNetPlusPlus(nn.Module):
    """Nested dense-skip U-Net++ implementation with the same encoder widths."""

    def __init__(self, in_channels=3, out_channels=1, filters: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        widths = list(map(int, filters)) + [int(filters[-1])]
        self.widths = widths
        self.depth = len(widths) - 1
        self.pool = nn.MaxPool2d(2)
        self.enc = nn.ModuleList()
        ch = in_channels
        for f in widths:
            self.enc.append(ConvBlock(ch, f)); ch = f
        self.nodes = nn.ModuleDict()
        for j in range(1, self.depth + 1):
            for i in range(self.depth - j + 1):
                in_ch = widths[i] * j + widths[i + 1]
                self.nodes[f"{i}_{j}"] = ConvBlock(in_ch, widths[i])
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        grid = {}
        for i, enc in enumerate(self.enc):
            x = enc(x); grid[(i, 0)] = x
            if i < self.depth:
                x = self.pool(x)
        for j in range(1, self.depth + 1):
            for i in range(self.depth - j + 1):
                parts = [grid[(i, k)] for k in range(j)]
                parts.append(upsample(grid[(i + 1, j - 1)], size=grid[(i, 0)].shape[-2:]))
                grid[(i, j)] = self.nodes[f"{i}_{j}"](torch.cat(parts, dim=1))
        return self.head(grid[(0, self.depth)])


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__()
        fs = tuple(map(int, filters)); self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList(); ch = in_channels
        for f in fs:
            self.encoders.append(ConvBlock(ch, f)); ch = f
        self.bridge = ConvBlock(fs[-1], fs[-1]); ch = fs[-1]
        self.gates = nn.ModuleList(); self.decoders = nn.ModuleList()
        for f in reversed(fs):
            self.gates.append(AttentionGate(f, ch, max(f // 2, 8)))
            self.decoders.append(ConvBlock(ch + f, f)); ch = f
        self.head = nn.Conv2d(ch, out_channels, 1)

    def forward(self, x):
        skips=[]
        for enc in self.encoders:
            x=enc(x); skips.append(x); x=self.pool(x)
        x=self.bridge(x)
        for gate, dec, skip in zip(self.gates, self.decoders, reversed(skips)):
            x=upsample(x, size=skip.shape[-2:]); s=gate(skip,x); x=dec(torch.cat([x,s],dim=1))
        return self.head(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters: Sequence[int] = (32, 64, 128)) -> None:
        super().__init__(); fs=tuple(map(int,filters)); self.pool=nn.MaxPool2d(2)
        self.encoders=nn.ModuleList(); ch=in_channels
        for f in fs: self.encoders.append(ResidualBlock(ch,f)); ch=f
        self.bridge=ResidualBlock(fs[-1],fs[-1]); ch=fs[-1]
        self.decoders=nn.ModuleList()
        for f in reversed(fs): self.decoders.append(ResidualBlock(ch+f,f)); ch=f
        self.head=nn.Conv2d(ch,out_channels,1)

    def forward(self,x):
        skips=[]
        for enc in self.encoders: x=enc(x); skips.append(x); x=self.pool(x)
        x=self.bridge(x)
        for dec,skip in zip(self.decoders,reversed(skips)):
            x=upsample(x,size=skip.shape[-2:]); x=dec(torch.cat([x,skip],dim=1))
        return self.head(x)


class ResUNetPlusPlus(nn.Module):
    """Residual + SE + ASPP + attention-gated decoder comparison model."""
    def __init__(self,in_channels=3,out_channels=1,filters:Sequence[int]=(32,64,128)) -> None:
        super().__init__(); fs=tuple(map(int,filters)); self.pool=nn.MaxPool2d(2)
        self.encoders=nn.ModuleList(); self.se=nn.ModuleList(); ch=in_channels
        for f in fs:
            self.encoders.append(ResidualBlock(ch,f)); self.se.append(SEBlock(f)); ch=f
        self.bridge=ASPP(fs[-1],fs[-1]); ch=fs[-1]
        self.gates=nn.ModuleList(); self.decoders=nn.ModuleList()
        for f in reversed(fs):
            self.gates.append(AttentionGate(f,ch,max(f//2,8)))
            self.decoders.append(ResidualBlock(ch+f,f)); ch=f
        self.head=nn.Conv2d(ch,out_channels,1)

    def forward(self,x):
        skips=[]
        for enc,se in zip(self.encoders,self.se): x=se(enc(x)); skips.append(x); x=self.pool(x)
        x=self.bridge(x)
        for gate,dec,skip in zip(self.gates,self.decoders,reversed(skips)):
            x=upsample(x,size=skip.shape[-2:]); s=gate(skip,x); x=dec(torch.cat([x,s],dim=1))
        return self.head(x)


class UNet3Plus(nn.Module):
    """Full-scale multi-level fusion decoder inspired by UNet 3+."""
    def __init__(self,in_channels=3,out_channels=1,filters:Sequence[int]=(32,64,128)) -> None:
        super().__init__(); fs=tuple(map(int,filters)); self.fs=fs; self.pool=nn.MaxPool2d(2)
        self.encoders=nn.ModuleList(); ch=in_channels
        for f in fs: self.encoders.append(ConvBlock(ch,f)); ch=f
        self.bridge=ConvBlock(fs[-1],fs[-1])
        cat_ch=fs[0]; self.proj=nn.ModuleDict(); self.fuse=nn.ModuleDict()
        # Decoder targets are encoder levels from deep to shallow.
        sources=list(fs)+[fs[-1]]
        for level in reversed(range(len(fs))):
            for s,ch_s in enumerate(sources):
                self.proj[f"{level}_{s}"]=nn.Sequential(nn.Conv2d(ch_s,cat_ch,1,bias=False),nn.BatchNorm2d(cat_ch),nn.ReLU(inplace=True))
            self.fuse[str(level)]=ConvBlock(cat_ch*len(sources),fs[level])
        self.head=nn.Conv2d(fs[0],out_channels,1)

    def forward(self,x):
        enc=[]
        for e in self.encoders: x=e(x); enc.append(x); x=self.pool(x)
        bridge=self.bridge(x); sources=enc+[bridge]; dec=None
        for level in reversed(range(len(self.fs))):
            target=enc[level].shape[-2:]; aligned=[]
            for s,feat in enumerate(sources):
                y=feat
                if y.shape[-2:]!=target:
                    y=F.adaptive_max_pool2d(y,target) if y.shape[-2]>target[0] else upsample(y,size=target)
                aligned.append(self.proj[f"{level}_{s}"](y))
            dec=self.fuse[str(level)](torch.cat(aligned,dim=1))
            sources[level]=dec
        return self.head(dec)


MODEL_NAMES = (
    "unet",
    "portable_bridge_legacy",
    "portable_bridge",
    "apb_unet",
    "unetpp",
    "attention_unet",
    "resunet",
    "resunetpp",
    "unet3plus",
)


def build_model(name: str, in_channels: int = 3, out_channels: int = 1, filters: Sequence[int] = (32,64,128)) -> nn.Module:
    key=name.lower().replace("-","_").replace("++","pp").replace("+","plus")
    aliases={
        "baseline":"unet","baseline_unet":"unet",
        "pb_unet":"portable_bridge","portable_bridge_unet":"portable_bridge",
        "pb_legacy":"portable_bridge_legacy",
        "adaptive_portable_bridge":"apb_unet","apb":"apb_unet",
        "u_netpp":"unetpp","unet_plus_plus":"unetpp",
        "attentionunet":"attention_unet","attunet":"attention_unet",
        "res_unet":"resunet","res_unetpp":"resunetpp","resunetplusplus":"resunetpp",
        "unet3p":"unet3plus","unet_3plus":"unet3plus",
    }
    key=aliases.get(key,key)
    table={
        "unet":BaselineUNet,
        "portable_bridge_legacy":PortableBridgeLegacyUNet,
        "portable_bridge":PortableBridgeUNet,
        "apb_unet":AdaptivePortableBridgeUNet,
        "unetpp":UNetPlusPlus,
        "attention_unet":AttentionUNet,
        "resunet":ResUNet,
        "resunetpp":ResUNetPlusPlus,
        "unet3plus":UNet3Plus,
    }
    if key not in table: raise ValueError(f"unknown model: {name}. Available: {', '.join(MODEL_NAMES)}")
    return table[key](in_channels,out_channels,filters)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
