from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("smooth must be finite and positive")
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs, targets = torch.sigmoid(logits.float()), targets.float()
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted BCE + soft Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        total = float(bce_weight) + float(dice_weight)
        if any(not math.isfinite(w) or w < 0 for w in (bce_weight, dice_weight)) or total <= 0:
            raise ValueError("bce_weight + dice_weight must be > 0")
        self.bce_weight = float(bce_weight) / total
        self.dice_weight = float(dice_weight) / total
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class FocalLoss(nn.Module):
    """Binary focal loss on logits."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


def build_loss(name: str) -> nn.Module:
    key = str(name).lower().replace("-", "_").replace("+", "_").replace(" ", "")
    aliases = {
        "bcewithlogitsloss": "bce",
        "bce": "bce",
        "dice": "dice",
        "diceloss": "dice",
        "bce_dice": "bce_dice",
        "bcedice": "bce_dice",
        "bcediceloss": "bce_dice",
        "focal": "focal",
        "focalloss": "focal",
    }
    key = aliases.get(key, key)
    if key == "bce":
        return nn.BCEWithLogitsLoss()
    if key == "dice":
        return DiceLoss()
    if key == "bce_dice":
        return BCEDiceLoss()
    if key == "focal":
        return FocalLoss()
    raise ValueError(f"unknown loss: {name}")
