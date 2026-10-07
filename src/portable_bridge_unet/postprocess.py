from __future__ import annotations

import cv2
import numpy as np


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask using flood fill."""
    x = (np.asarray(binary).squeeze() > 0).astype(np.uint8) * 255
    h, w = x.shape
    flood = x.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(x, flood_inv)


def largest_component(binary: np.ndarray) -> np.ndarray:
    x = (np.asarray(binary).squeeze() > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(x, connectivity=8)
    if n <= 1:
        return x.astype(np.uint8) * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype(np.uint8) * 255


def postprocess(binary: np.ndarray) -> np.ndarray:
    """Paper-code equivalent: hole filling + largest connected component."""
    return largest_component(fill_holes(binary))
