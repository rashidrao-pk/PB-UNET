"""Post-processing corresponding to the original implementation.

The actual legacy code performs morphological hole filling by reconstruction,
then keeps the largest connected component. It does *not* estimate missing
pixels from the average contour intensity.
"""
import numpy as np
from skimage.morphology import reconstruction
from skimage.measure import label, regionprops


def fill_holes_keep_largest(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    mask = (mask > 0).astype(np.uint8)
    if mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    seed = mask.copy()
    seed[1:-1, 1:-1] = mask.max()
    filled = reconstruction(seed, mask, method="erosion")

    labeled = label(filled)
    regions = regionprops(labeled.astype(int))
    if not regions:
        return np.zeros_like(mask, dtype=np.uint8)
    region = max(regions, key=lambda r: r.area)

    out = np.zeros_like(mask, dtype=np.uint8)
    minr, minc, maxr, maxc = region.bbox
    out[minr:maxr, minc:maxc] = region.filled_image.astype(np.uint8)
    return out
