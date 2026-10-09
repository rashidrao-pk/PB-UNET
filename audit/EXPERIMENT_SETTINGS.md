# Experiment settings audit — 2026-10-09

The checkout initially retained only BUSI/Kvasir benchmark configs and one Kvasir
multi-seed config. A preceding commit had removed the earlier data library,
BraTS/DRIVE/ISIC/Montgomery support and regression tests. These were recovered
from commit `64e357a`, preserving the newer profiling, qualitative, multi-seed
and CVC experiment capabilities, then integrated with the suite runner.

| Dataset | PB/U-Net settings | Nine-model benchmark | Three training seeds | Current protocol |
|---|---|---|---|---|
| Sunnybrook | Local/Epito | Local/Epito | Local/Epito | Explicit patient-grouped grayscale split |
| BUSI | Local/Epito | Local/Epito | Local/Epito | Existing prepared lesion split CSVs |
| Kvasir-SEG | Local/Epito | Local/Epito | Local/Epito | Shared image split CSVs |
| Montgomery | Local/Epito | Local/Epito | Local/Epito | Both lungs; image split |
| ISIC 2016 | Local/Epito | Local/Epito | Local/Epito | Official test; train-byte overlap exclusion |
| BraTS 2020 | Local/Epito | Local/Epito | Local/Epito | 2D FLAIR WT, case split before slices |
| BraTS 2021 | Local/Epito | Local/Epito | Local/Epito | Separate 2D FLAIR WT case split |
| DRIVE | Local/Epito | Local/Epito | Local/Epito | Official test, first manual annotator, FOV |
| CVC-ClinicDB | External test settings | Source Kvasir checkpoints | Each source seed supported | Entire cohort external; no target training |

All eight training datasets use explicit splits, nine supported model names,
matched config channels and resize settings, BCE+Dice, seed 42 for a single run,
and seeds 42/123/2026 for repeated training. Existing historical configs remain
separate and do not automatically establish a patient-grouped experiment.

No complete benchmark summary/leaderboard was found in the local `runs/` tree
at the start of this audit. This does not describe files on Epito. A settings
inventory is not evidence of completed training or model quality. Run
`check_experiment_inventory.py` on the machine holding the actual artifacts.

Correctness checks include restored preparation/architecture/checkpoint/FOV
tests, suite-config consistency, no-write dry runs, overlapping split rejection,
preflight failure handling, grouped split protection, and synthetic end-to-end
single/multi-seed execution. Real dataset preparation, complete GPU convergence,
external validity and manuscript results remain to be verified on the target
machine. Use [the run guide](../docs/BENCHMARKS.md) for commands and limitations.

## Verification completed

- 92 regression tests passed in the existing isolated Python 3.10 environment.
- That environment's `pip check` passed with no broken requirements.
- All 36 local/Epito single-seed, multi-seed and external child-runner dry runs passed.
- Synthetic single- and multi-seed suites completed training, checkpoints,
  evaluations, comparison/summary generation and persistent suite reports.
- A synthetic four-checkpoint external evaluation completed without fine-tuning.
- Local Markdown file targets and Python syntax were checked; `git diff --check` passed.
- [Local settings/artifact inventory](EXPERIMENT_INVENTORY_LOCAL.json) found no
  complete benchmark evaluation artifacts at the configured standard/suite paths.

The author's shared `pt` environment fails `pip check` due to OpenCV 4.7 below
the declared >=4.8 minimum and unrelated Captum/SAM NumPy conflicts. Those
packages were not changed for this audit. Use an isolated compatible environment
on Epito, then run `pip check` and the tests there. No GPU or complete real-data
benchmark was launched or verified during this audit.
