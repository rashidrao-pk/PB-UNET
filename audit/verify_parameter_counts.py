#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portable_bridge_unet.models import BaselineUNet, PortableBridgeLegacyUNet, count_trainable_parameters

EXPECTED = {
    "BaselineUNet": 1_211_649,
    "PortableBridgeLegacyUNet": 2_356_609,
}

for cls in (BaselineUNet, PortableBridgeLegacyUNet):
    model = cls()
    n = count_trainable_parameters(model)
    print(f"{cls.__name__}: {n:,} trainable parameters")
    assert n == EXPECTED[cls.__name__], (cls.__name__, n, EXPECTED[cls.__name__])

print("OK: PyTorch trainable counts exactly match the manuscript/Keras trainable counts.")
print("Keras 'total' is larger because it also counts BatchNorm running mean/variance as non-trainable values.")
