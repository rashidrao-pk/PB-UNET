```bash
mkdir -p \
/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw
```




```bash
hf download berkaytrhn/cvc-clinicdb \
  --repo-type dataset \
  --local-dir \
  /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw
```


```bash
find \
/beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw \
-maxdepth 3 -type d | sort
```


```bash
python scripts/prepare_cvc_clinicdb.py \
  --raw-root \
  /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw \
  --prepared-root \
  /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/cvc_clinicdb \
  --dry-run
```


```bash
python scripts/evaluate.py \
  --checkpoint \
  runs/benchmark_kvasir/unet/best_dice.pt \
  --split-csv \
  /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/cvc_clinicdb/metadata/test_split.csv \
  --out-dir \
  runs/external_cvc/unet
```

```bash
python scripts/evaluate.py \
  --checkpoint \
  runs/benchmark_kvasir/portable_bridge_legacy/best_dice.pt \
  --split-csv \
  /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/cvc_clinicdb/metadata/test_split.csv \
  --out-dir \
  runs/external_cvc/portable_bridge_legacy
```