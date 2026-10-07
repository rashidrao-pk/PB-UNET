"""Publication-oriented full-test-set evaluation."""
from __future__ import annotations
import numpy as np


def dice_np(y_true, y_pred, eps=1e-7):
    a = np.asarray(y_true).astype(bool)
    b = np.asarray(y_pred).astype(bool)
    return (2 * np.logical_and(a, b).sum() + eps) / (a.sum() + b.sum() + eps)


def iou_np(y_true, y_pred, eps=1e-7):
    a = np.asarray(y_true).astype(bool)
    b = np.asarray(y_pred).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter + eps) / (union + eps)
