```bash
/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/
```


```bash
cd /beegfs/home/mrashid/repos/PB-UNET

ssh epito-mercurio

# Check all Available sources
sinfo -N -p mirri,gracehopper,cascadelake,epito \
  -o "%.18N %.18P %.10T %.16G %.20C"

# Check all Reserved sources
squeue -p mirri,gracehopper,cascadelake,epito \
  -o "%.12i %.12u %.18P %.18j %.8T %.15N %.12b %.20R"

srun -p epito --gres=gpu:a100:1 -J "PB-UNET" --pty bash
tmux new -s PBUNET
source /beegfs/home/mrashid/pt_312/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages/
```

```bash
cd /beegfs/home/mrashid/repos/PB-UNET

python scripts/check_dataset.py --config configs/paper_legacy_epito.yaml
python scripts/prepare_sunnybrook.py --config configs/paper_legacy_epito.yaml
python scripts/check_dataset.py --config configs/paper_legacy_epito.yaml
```


Train PB-U-Net:

```bash
python scripts/train.py \
  --config configs/busi_pb_epito.yaml \
  --train-split /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/BUSI/metadata/train_split.csv \
  --val-split /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/BUSI/metadata/val_split.csv \
  --test-split /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/BUSI/metadata/test_split.csv
```

Train the baseline:

```bash
python scripts/train.py --config configs/baseline_epito.yaml
```

## Evaluate 
- Run the saved best checkpoint on the full test split:

```bash
python scripts/evaluate.py \
  --checkpoint runs/pb_unet/best.pt \
  --split-csv runs/pb_unet/test_split.csv \
  --out-dir runs/pb_unet/evaluation
```

- Then run the same checkpoint with post-processing:

```bash
python scripts/evaluate.py \
  --checkpoint runs/pb_unet/best.pt \
  --split-csv runs/pb_unet/test_split.csv \
  --postprocess \
  --out-dir runs/pb_unet/evaluation_postprocessed
```

### Evaluate UNet

```bash
python scripts/evaluate.py \
  --checkpoint runs/unet/best.pt \
  --split-csv runs/unet/test_split.csv \
  --out-dir runs/unet/evaluation
```

#### With Post Processing
```bash
python scripts/evaluate.py \
  --checkpoint runs/unet/best.pt \
  --split-csv runs/unet/test_split.csv \
  --postprocess \
  --out-dir runs/unet/evaluation_postprocessed
```


## DOWNLOAD BUSI Dataset

```bash
mkdir -p /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/BUSI/raw

python - <<'PY'
import xxhash
print("xxhash OK:", xxhash.xxh64(b"test").hexdigest())

import datasets
import huggingface_hub

print("datasets:", datasets.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
PY

```


```bash
python - <<'PY'
from datasets import load_dataset
from collections import Counter
import numpy as np

CACHE = "/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/BUSI/raw"

ds = load_dataset(
    "MedOtter/BUSI",
    split="train",
    cache_dir=CACHE,
)

print("\n=== DATASET ===")
print(ds)
print("\n=== FEATURES ===")
print(ds.features)
print("\nTotal rows:", len(ds))

print("\n=== FIRST SAMPLE KEYS ===")
print(ds[0].keys())

# class distribution
labels = [x["class_label"] for x in ds]
print("\n=== CLASS DISTRIBUTION ===")
print(Counter(labels))

print("\n=== FIRST SAMPLE ===")
sample = ds[0]

for k, v in sample.items():
    if k not in ["image", "mask"]:
        print(k, ":", v)

img = sample["image"]
mask = sample["mask"]

print("\nImage mode :", img.mode)
print("Image size :", img.size)

print("Mask mode  :", mask.mode)
print("Mask size  :", mask.size)

img_np = np.array(img)
mask_np = np.array(mask)

print("\nImage shape:", img_np.shape)
print("Image dtype:", img_np.dtype)
print("Image range:", img_np.min(), img_np.max())

print("\nMask shape :", mask_np.shape)
print("Mask dtype :", mask_np.dtype)
print("Mask unique:", np.unique(mask_np))
PY
```

```bash
python - <<'PY'
from datasets import load_dataset
from collections import Counter
import numpy as np

CACHE = "/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/BUSI/raw"

ds = load_dataset(
    "MedOtter/BUSI",
    split="train",
    cache_dir=CACHE,
)

bad_size = []
bad_mask_values = []
empty_masks = []
nonempty_masks = []

for i, sample in enumerate(ds):
    label = sample["class_label"]

    img = np.array(sample["image"])
    mask = np.array(sample["mask"])

    if img.shape[:2] != mask.shape[:2]:
        bad_size.append(
            (i, sample["image_id"], img.shape, mask.shape)
        )

    unique = np.unique(mask)

    if not set(unique.tolist()).issubset({0, 255}):
        bad_mask_values.append(
            (i, sample["image_id"], unique.tolist())
        )

    if mask.max() == 0:
        empty_masks.append((i, label, sample["image_id"]))
    else:
        nonempty_masks.append((i, label, sample["image_id"]))

print("\n=== BUSI FULL VERIFICATION ===")

print("Total:", len(ds))
print("Class distribution:")
print(Counter(x["class_label"] for x in ds))

print("\nBad image/mask sizes:", len(bad_size))
print("Bad mask values:", len(bad_mask_values))

print("\nEmpty masks:", len(empty_masks))
print("Empty mask classes:")
print(Counter(x[1] for x in empty_masks))

print("\nNon-empty masks:", len(nonempty_masks))
print("Non-empty mask classes:")
print(Counter(x[1] for x in nonempty_masks))

if bad_size:
    print("\nFirst bad size examples:")
    for x in bad_size[:10]:
        print(x)

if bad_mask_values:
    print("\nFirst bad mask examples:")
    for x in bad_mask_values[:10]:
        print(x)
PY
```


```bash
python - <<'PY'
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

ROOT = Path(
    "/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/BUSI"
)

df = pd.read_csv(ROOT / "metadata" / "samples_lesions.csv")

train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class"],
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["class"],
)

OUT = ROOT / "metadata"

train.to_csv(OUT / "train_split.csv", index=False)
val.to_csv(OUT / "val_split.csv", index=False)
test.to_csv(OUT / "test_split.csv", index=False)

print("\n=== SPLIT COUNTS ===")

for name, split in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    print(f"\n{name}: {len(split)}")
    print(split["class"].value_counts())
PY
```


```bash
python - <<'PY'
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

ROOT = Path(
    "/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/BUSI"
)

df = pd.read_csv(ROOT / "metadata" / "samples_lesions.csv")

train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class"],
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["class"],
)

OUT = ROOT / "metadata"

train.to_csv(OUT / "train_split.csv", index=False)
val.to_csv(OUT / "val_split.csv", index=False)
test.to_csv(OUT / "test_split.csv", index=False)

print("\n=== SPLIT COUNTS ===")

for name, split in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    print(f"\n{name}: {len(split)}")
    print(split["class"].value_counts())
PY
```