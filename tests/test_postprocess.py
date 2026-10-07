import numpy as np
from portable_bridge_unet.postprocess import postprocess


def test_postprocess_keeps_largest_and_fills_hole():
    x = np.zeros((32, 32), dtype=np.uint8)
    x[4:20, 4:20] = 255
    x[9:12, 9:12] = 0
    x[25:28, 25:28] = 255
    y = postprocess(x)
    assert y[10, 10] == 255
    assert y[26, 26] == 0
