# Reproducibility protocol

This file describes the executable pipeline and the artifacts needed to reproduce
an experiment. It does not certify results that have not been run. Publication
requirements and outstanding evidence are tracked in
[the readiness audit](docs/PUBLICATION_READINESS.md).

## Freeze the experiment before testing

Record the dataset release, target, source, access terms, exclusions, and available
patient identifiers. Choose splits, preprocessing, primary metric, model selection
criterion, statistical unit, training seeds, and compute budget before inspecting
test scores. Use the same policy across compared models. Hyperparameters,
thresholds, and postprocessing are chosen using training/validation data only.
The test set is evaluated after model/protocol choices are frozen.

For each training seed, use a new output directory. For multi-seed architecture
comparisons, keep the data split fixed and vary training initialization seeds
explicitly. The benchmark currently accepts one training seed per invocation;
change `training.seed` and `out_root` in a copy of its config for each run.
Do not confuse bootstrap resampling seeds with training seeds.

## Environment and installation

Install from a recorded Git revision using Python >=3.10 and a PyTorch build
appropriate for the target accelerator:

```bash
python -m pip install -e '.[test]'
python -m pip check
python -m pytest -q
python audit/verify_parameter_counts.py
python -m pip freeze > experiment-environment.txt
```

`pyproject.toml` specifies supported lower bounds, not an exact environment lock.
Preserve the resolved package list, Python version, GPU/driver version, CUDA
runtime, operating system, and installation commands with the experiment.
A frozen list from macOS cannot be assumed to reproduce a Linux CUDA environment.
A [tested macOS CPU environment](environments/README.md) is supplied.
No multi-platform environment or GPU reproducibility claim is implied by local CPU tests. A release-specific container/environment should be verified
and archived alongside final paper results.

## Data provenance and splits

Keep raw files immutable and retain preparer manifests, original source hashes
where provided, preparation reports, and split CSVs. Scripts do not redistribute
raw data. Data paths are machine-specific; their contents and identities define
the experiment. CSV image/mask paths are currently literal paths, so migrating
machines requires consistent path rewriting, followed by identity/hash checks.

| Dataset | Current split policy | Material limitation |
|---|---|---|
| Sunnybrook legacy | Image-level random split | Frames from a patient may cross splits; historical reproduction only |
| Sunnybrook grouped configs | Patient `group` in preparation manifest | Use grouped configs for primary new patient-level claims |
| BUSI | Explicit prepared split CSVs | Patient independence must be audited from the actual source metadata |
| Kvasir-SEG | Seeded image split | Not verified patient/video independent |
| Montgomery | Seeded stratified image split | Numeric case IDs do not prove independent patient linkage |
| ISIC 2016 | Official test retained; train-byte duplicates excluded, then validation holdout | Image split; not verified lesion/patient independent |
| BraTS 2020/2021 | Case split before axial slicing | Internal split of labeled training release; do not merge releases as independent cohorts |
| DRIVE | Official test, validation holdout from training | First manual annotator, FOV domain; patient linkage unavailable |

A patient-grouped Sunnybrook experiment can use the supplied
`configs/sunnybrook_grouped_pb.yaml` / `sunnybrook_grouped_unet.yaml` (or `_epito`)
after source preparation. The trainer supports `--manifest-csv` and
`--split-unit group`; it rejects missing or overlapping groups. The grouped
configs use seed 42 for both data split and initialization. For initialization-only
reruns, reuse the first run's explicit train/val/test CSVs and vary `--seed`.
Never resplit images independently for each architecture.

## Preprocessing and evaluation

The loader resizes images bilinearly and binary masks with nearest-neighbor
interpolation, converts images to float32 /255, and thresholds masks at >0.5.
Channels, size and augmentation are set by config. Source preparation differs
by dataset; cite the corresponding dataset guide and saved preparation metadata.
BraTS uses per-volume nonzero-voxel percentile normalization before uint8 export;
this is a 2D single-modality protocol, not the full 3D challenge pipeline.

Prediction uses sigmoid probabilities and the configured threshold (default
0.5). Report mean, standard deviation and median across the complete evaluation
set. Dice/IoU/precision/recall/accuracy use binary confusion counts. Epsilon
smoothing yields 1 for empty/empty foreground comparisons, including undefined
precision/recall denominators; report empty-mask prevalence and this convention.
HD95 uses symmetric surface distances in **resized pixels**, not physical units.
Both masks empty gives 0; exactly one empty gives the image diagonal. These
fallback distances differ from some published challenge conventions.

DRIVE metrics count only pixels within the supplied FOV, including accuracy;
training loss remains full-image. Retain the `roi` CSV column. Morphological
postprocessing fills holes then keeps one largest connected component. Use raw
outputs as the primary comparison and select any postprocessing policy on
validation data. Largest-component filtering can discard valid multiple organs,
lesions or branches, so it is disabled in relevant supplied configs.

`best_dice.pt` is selected using validation Dice; `best_loss.pt` uses validation
loss. Learning-rate reduction and early stopping use validation loss. Declare
one checkpoint policy in advance; do not choose it using test performance.
Checkpoints store model/config metadata and weights, but do not include a complete
optimizer/RNG/scheduler state for exact training resumption.

## Randomness

`--seed` seeds Python, NumPy and PyTorch. `--deterministic` requires deterministic
PyTorch algorithms, disables cuDNN benchmarking, enables deterministic cuDNN,
and sets a cuBLAS workspace setting before CUDA initialization. Unsupported
operations raise an error rather than silently relaxing determinism. In benchmark
configs set `training.deterministic: true` to forward the flag.
Determinism can reduce speed and does not guarantee identical outputs across
hardware, PyTorch versions or operating systems. Without this flag, cuDNN
benchmarking is enabled. Record the actual settings in provenance.

## Artifacts generated automatically

Training `provenance.json` and evaluation `provenance_raw.json` /
`provenance_postprocessed.json` capture:

- UTC timestamp, actual command and resolved settings;
- Git commit and working-tree status when available;
- hashes of Python source files, dependency declarations, and supplied inputs;
- exact installed package versions and companion `*_packages.txt`;
- Python/platform, selected device, CUDA/cuDNN and GPU names;
- deterministic-algorithm and cuDNN settings.

Training hashes saved split CSVs and supplied preparation manifests/reports;
evaluation additionally hashes the evaluated checkpoint and test CSV. These
hashes identify artifacts; they do not snapshot image contents or guarantee
raw-data availability. Preserve actual sources, manifests, and prepared data
checksums with the paper. A dirty Git tree requires archiving the exact code,
not merely reporting the commit. Source hashes detect changes but cannot recover
missing source code. No credentials or environment-variable dump is collected.
Paths and device metadata may still identify local infrastructure; review them
before public archiving.

## Publication artifact checklist

- [ ] Fixed Git tag/commit and archived code release with persistent identifier.
- [ ] Verified environment specification/container, GPU driver information.
- [ ] Dataset access and citation statements, exact releases and exclusions.
- [ ] Split CSVs and patient/case independence audit.
- [ ] Configs, provenance, preparation manifests and smoke inspection reports.
- [ ] Histories, selected checkpoints, complete per-image outputs and plots.
- [ ] Repeated training runs with declared seeds and variability.
- [ ] Ablations, contemporary verified baselines, matched compute/tuning budgets.
- [ ] Patient/case-aware uncertainty where relevant; multiple-comparison plan.
- [ ] Independent reproduction of the main table from archived artifacts.

Do not upload restricted data or patient metadata merely to complete a checklist.
State access restrictions and provide reproducible preparation steps instead.
