```bash
python - <<'PY'
import torch
import cv2
import numpy as np
import pandas as pd
import sklearn
import scipy

print("torch:", torch.__version__)
print("opencv:", cv2.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("scipy:", scipy.__version__)
PY
```

```bash
python - <<'PY'
from portable_bridge_unet.models import BaselineUNet, PortableBridgeUNet

u = BaselineUNet(in_channels=3)
p = PortableBridgeUNet(in_channels=3)

print("Baseline trainable params:",
      sum(x.numel() for x in u.parameters() if x.requires_grad))

print("PB-U-Net trainable params:",
      sum(x.numel() for x in p.parameters() if x.requires_grad))
PY
```

- After the parameter check, run a forward-pass test too:

```bash
python - <<'PY'
import torch
from portable_bridge_unet.models import BaselineUNet, PortableBridgeUNet

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print("Device:", device)

x = torch.randn(2, 3, 256, 256).to(device)

for Model in [BaselineUNet, PortableBridgeUNet]:
    model = Model(in_channels=3).to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)

    print(f"{Model.__name__}:")
    print("  input :", tuple(x.shape))
    print("  output:", tuple(y.shape))
    print("  params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
PY
```

## Configuration and data checks

Run from the repository root. Active scripts default to
`configs/paper_legacy.yaml`; `configs/baseline.yaml` selects the baseline run.
Dataset paths belong in the YAML `dataset` section, so they do not need to be
repeated in commands. See [training](TRAIN.md) and [evaluation](EVALUATE.md).

```bash
python scripts/check_dataset.py
python scripts/prepare_sunnybrook.py
python scripts/check_dataset.py --require-prepared
python -m pytest -q
```

The raw dataset is present at the configured location. The preparation script
creates PNG image/mask pairs before checking training readiness.

- Upload Dataset to Server

```bash
cd /Users/rashid/data/DS/Healthcare
zip -r PB_U-NET.zip PB_U-NET -x '*/.DS_Store'

scp /Users/rashid/data/DS/SR/v6/Jul27/test.zip mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/datasets/Healthcare

# Upload test data

unzip -o /beegfs/home/mrashid/datasets/AD/SR/V6/test.zip \
  -d /beegfs/home/mrashid/datasets/AD/SR/V6/test \
  -x "__MACOSX/*"

```
