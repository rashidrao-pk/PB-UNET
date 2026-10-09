# Final experiments for the cross-domain PB-U-Net paper

## 1. CVC-ClinicDB external evaluation

Download the dataset according to its research-use terms. A convenient mirror is available on Hugging Face, but the original dataset terms remain applicable.

Example on Epito:

```bash
mkdir -p /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw
hf download berkaytrhn/cvc-clinicdb \
  --repo-type dataset \
  --local-dir /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw
```

Inspect:

```bash
find /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw -maxdepth 3 -type d | sort
```

Prepare all pairs as one external test set:

```bash
python scripts/prepare_cvc_clinicdb.py \
  --raw-root /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw \
  --prepared-root /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/cvc_clinicdb \
  --dry-run

python scripts/prepare_cvc_clinicdb.py \
  --raw-root /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/CVC-ClinicDB/raw \
  --prepared-root /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/cvc_clinicdb
```

Evaluate Kvasir-trained checkpoints without fine-tuning:

```bash
python scripts/evaluate.py \
  --checkpoint runs/benchmark_kvasir/unet/best_dice.pt \
  --split-csv /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/cvc_clinicdb/metadata/test_split.csv \
  --out-dir runs/external_cvc/unet

python scripts/evaluate.py \
  --checkpoint runs/benchmark_kvasir/portable_bridge_legacy/best_dice.pt \
  --split-csv /beegfs/home/mrashid/datasets/Healthcare/PB_U-NET/data/cvc_clinicdb/metadata/test_split.csv \
  --out-dir runs/external_cvc/portable_bridge_legacy
```

Then create a comparison config or point `compare_models.py` to these two per-image CSVs.

## 2. Three-seed Kvasir experiment

```bash
python scripts/run_multiseed.py --config configs/multiseed_kvasir_epito.yaml
```

Outputs:
- `runs/multiseed_kvasir/per_seed_metrics.csv`
- `runs/multiseed_kvasir/multiseed_summary.csv`

## 3. Complexity and inference profiling

Run on the same Epito GPU you will report in the article:

```bash
python scripts/profile_models.py \
  --image-size 256 \
  --batch-size 1 \
  --warmup 50 \
  --iters 200 \
  --device cuda \
  --out runs/profiling/model_profile_bs1.csv
```

Optional throughput benchmark with batch 8:

```bash
python scripts/profile_models.py \
  --image-size 256 \
  --batch-size 8 \
  --warmup 50 \
  --iters 200 \
  --device cuda \
  --out runs/profiling/model_profile_bs8.csv
```

The FLOP estimate counts Conv2d/Linear multiply-add as 2 FLOPs. State this convention in the paper.

## 4. Qualitative PB-vs-U-Net cases

```bash
python scripts/select_qualitative_cases.py \
  --pb-csv runs/benchmark_kvasir/portable_bridge_legacy/evaluation/per_image_raw.csv \
  --unet-csv runs/benchmark_kvasir/unet/evaluation/per_image_raw.csv \
  --pb-checkpoint runs/benchmark_kvasir/portable_bridge_legacy/best_dice.pt \
  --unet-checkpoint runs/benchmark_kvasir/unet/best_dice.pt \
  --n 5 \
  --device cuda \
  --out-dir runs/qualitative_kvasir
```

This creates:
- `pb_wins.png`
- `comparable.png`
- `pb_failures.png`
- `selected_cases.csv`

The cases are selected algorithmically from paired Dice differences, reducing manual cherry-picking.

## Unified suite

See [BENCHMARKS.md](BENCHMARKS.md) for all-dataset single/multi-seed execution, preparation preflight and source-checkpoint external evaluation. CVC can now be prepared with `--config configs/cvc_clinicdb_external_epito.yaml`.
