# Run every configured dataset experiment

The suite includes **eight training datasets** and one external evaluation:
Sunnybrook, BUSI, Kvasir-SEG, Montgomery, ISIC 2016, BraTS 2020, BraTS 2021,
DRIVE, and CVC-ClinicDB. Both local and Epito suites are configured.
Settings are available; their presence does not mean experiments have completed.
Remote Epito files and GPU runs cannot be verified from this local checkout.

## Inventory and dry-run

```bash
python scripts/check_experiment_inventory.py --config configs/benchmark_all_epito.yaml
python scripts/run_all_datasets.py --config configs/benchmark_all_epito.yaml --dry-run
python scripts/run_all_datasets.py --config configs/benchmark_all_epito.yaml --mode multiseed --dry-run
```

Inventory reports split-file availability on the machine where it runs and
counts evaluation artifacts in standard/suite output locations. A found summary
is not independently verified scientific evidence. Dry-run checks configuration
consistency and prints the command plan without reading datasets, training, or
creating output directories. It cannot establish dataset readiness.

## Prepare downloaded data and verify readiness

Acquire/extract the datasets using their respective guides. BraTS/DRIVE access
remains manual. `--prepare` never downloads data. Edit source/prepared/split
paths in each dataset's experiment, benchmark and multi-seed configs consistently.

```bash
python scripts/run_all_datasets.py \
  --config configs/benchmark_all_epito.yaml --prepare --check-only
```

Preparation runs only when required split files are missing. BraTS/DRIVE/CVC
preparers require an empty destination when conversion is needed; keep existing
protocols and use new destinations rather than mixing files. BUSI must already
have prepared lesion pairs and explicit train/val/test CSVs. The suite does not
invent a new BUSI patient split.

For Sunnybrook, the new suite uses a patient-grouped grayscale experiment:

```bash
python scripts/prepare_sunnybrook.py --config configs/sunnybrook_pb_epito.yaml
python scripts/prepare_sunnybrook_splits.py --config configs/sunnybrook_pb_epito.yaml
```

The second command creates shared train/val/test CSVs from the patient groups
in the preparation manifest. All models use those same CSVs. The historical
`paper_legacy` and `baseline` configs retain their separate image-level protocol.
Changed grouped splits cannot silently overwrite existing split CSVs.

Preflight validates nonempty splits, image decoding, duplicate/overlapping image
identities, complete prepared-image coverage, and required patient/case group
separation. DRIVE requires FOV paths. It also saves image/mask/overlay previews,
resolved experiment configs and model-forward smoke reports for every selected
dataset. All selected datasets are checked before any training starts; failures
are reported together in `suite_report.json` and the process exits nonzero.

## Full runs

All-model single-seed run:

```bash
python scripts/run_all_datasets.py --config configs/benchmark_all_epito.yaml
```

All-model three-seed run:

```bash
python scripts/run_all_datasets.py \
  --config configs/benchmark_all_epito.yaml --mode multiseed
```

Single-seed runs schedule 72 training jobs (8 datasets × 9 models), plus 9
external CVC evaluations. Three-seed runs schedule 216 training jobs and 27
external CVC evaluations. They execute sequentially; this is not a Slurm job
submission tool. Select datasets/models in configs to fit the compute budget.
The training seed, AMP and augmentation choices are explicitly forwarded.
Strict deterministic training is configured; unsupported deterministic GPU
operations can raise errors. Verify your actual accelerator/PyTorch combination;
change `training.deterministic` consistently if using a relaxed protocol.

Outputs use a fresh UTC timestamp under `runs/dataset_suite/`. Resolved benchmark
configs are copied into the run before execution. Child commands use those copies,
so editing the original YAML during a long run does not change later stages.
The suite report records the selected mode, commands, readiness, completion/failure
states and suite-config hash. Training protects existing checkpoint/history files.

Choose a subset, using canonical dataset names:

```bash
python scripts/run_all_datasets.py \
  --config configs/benchmark_all_epito.yaml --datasets busi kvasir_seg isic2016
```

To reevaluate an existing suite, supply its exact run directory:

```bash
python scripts/run_all_datasets.py \
  --config configs/benchmark_all_epito.yaml \
  --out-dir runs/dataset_suite/EXISTING_TIMESTAMP --skip-training
```

This reruns evaluation and summaries, not exact optimizer-state training resumption.
For local paths use `configs/benchmark_all.yaml`. Canonical multi-seed Kvasir local
settings are now `multiseed_kvasir.yaml`; cluster settings use `_epito` explicitly.

## Individual datasets

Each training dataset has these patterns:

```text
configs/<dataset>_pb[_epito].yaml
configs/<dataset>_unet[_epito].yaml
configs/benchmark_<dataset>[_epito].yaml
configs/multiseed_<dataset>[_epito].yaml
```

Use `kvasir` in filenames and `kvasir_seg` in suite selection. Examples:

```bash
python scripts/run_benchmark.py --config configs/benchmark_drive_epito.yaml
python scripts/run_multiseed.py --config configs/multiseed_brats2020_epito.yaml
```

Use `--out-root /path/to/new/run` for an independent run. Individual runners
validate explicit splits before execution. Dry runs write nothing. Multi-seed
summaries contain per-seed metrics and means/standard deviations across training
seeds; they do not substitute for patient/case-aware uncertainty.

## External CVC-ClinicDB

CVC is an external test set, not an additional in-domain training dataset:

```bash
python scripts/prepare_cvc_clinicdb.py --config configs/cvc_clinicdb_external_epito.yaml
python scripts/run_external_benchmark.py --config configs/benchmark_cvc_external_epito.yaml
```

The full suite evaluates CVC after Kvasir and automatically points at the
Kvasir checkpoints produced in that same suite run, including each seed in
multi-seed mode. No CVC fitting or fine-tuning occurs. If only CVC is selected,
the configured source checkpoint root must already exist. Standalone external
settings default to `runs/benchmark_kvasir`; override `--source-root` if needed.
All source checkpoints are checked before external evaluation starts.

## Scientific limits

Separately training each dataset establishes applicability, not cross-dataset
transfer. CVC is the explicitly configured external transfer test. BraTS is a
single-modality, binary 2D internal case holdout, not official 3D challenge
scoring; releases must not be merged as independent cohorts. Image-level splits
in datasets without patient/video linkage do not establish patient independence.
Published architecture names identify local implementations/adaptations.

See [dataset guides](DATASETS.md), [comparison protocol](MODEL_COMPARISONS.md),
and the current [experiment-settings audit](../audit/EXPERIMENT_SETTINGS.md).
