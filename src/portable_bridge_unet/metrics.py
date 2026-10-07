from __future__ import annotations

import numpy as np
import torch


def _flatten_binary(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def binary_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).to(torch.float32)


def batch_confusion(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
    pred = _flatten_binary(pred.float())
    target = _flatten_binary(target.float())
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    tn = ((1 - pred) * (1 - target)).sum(dim=1)
    return tp, fp, fn, tn


def batch_metrics_from_logits(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7
) -> dict[str, torch.Tensor]:
    pred = binary_predictions(logits, threshold)
    tp, fp, fn, tn = batch_confusion(pred, target)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
    }


def hd95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance in pixels.

    Uses scipy only when this metric is requested. Empty-mask behavior:
      both empty -> 0; exactly one empty -> image diagonal.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred = np.asarray(pred).astype(bool).squeeze()
    target = np.asarray(target).astype(bool).squeeze()
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        h, w = pred.shape[-2:]
        return float(np.hypot(h, w))

    pred_surface = pred ^ binary_erosion(pred)
    target_surface = target ^ binary_erosion(target)
    dt_target = distance_transform_edt(~target_surface)
    dt_pred = distance_transform_edt(~pred_surface)
    d1 = dt_target[pred_surface]
    d2 = dt_pred[target_surface]
    return float(np.percentile(np.concatenate([d1, d2]), 95))
