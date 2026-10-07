# Evaluation and prediction

Run commands from the repository root after training. Checkpoint, split CSV,
and output paths come from the selected YAML config.

Evaluate PB-U-Net:

```bash
python scripts/evaluate.py --config configs/paper_legacy.yaml
python scripts/evaluate.py --config configs/paper_legacy.yaml --postprocess
```

Evaluate the baseline:

```bash
python scripts/evaluate.py --config configs/baseline.yaml
python scripts/evaluate.py --config configs/baseline.yaml --postprocess
```

Evaluation uses `evaluate.checkpoint`, `evaluate.split_csv`, and
`evaluate.out_dir`. Architecture, input size, channels, and threshold are loaded
from the checkpoint. Set `evaluate.threshold` or pass `--threshold` to override
the saved threshold.

For single-image prediction, set `predict.image` in the chosen YAML to an
existing prepared image, then run:

```bash
python scripts/predict.py --config configs/paper_legacy.yaml --postprocess
```

Prediction uses `predict.checkpoint` and `predict.output` from YAML. The input
image is deliberately unset until you choose one. CLI options remain available
to override configured paths or settings.

With no `--config`, both scripts use `configs/paper_legacy.yaml`. Relative YAML
paths resolve against the config directory; relative CLI paths resolve against
the current working directory.

## Saved evaluation plots

Evaluation saves these plots next to the per-image CSV and summary JSON, in
both PNG and PDF formats:

- `metric_distributions_raw.*`: per-image score histograms, plus HD95 in pixels
  when computed.
- `metric_summary_raw.*`: mean scores with sample standard deviation and
  per-image boxplots.

Postprocessed evaluation uses the `_postprocessed` suffix. HD95 is plotted
separately from scores because it uses distance units. Histogram titles report
finite and nonfinite counts; nonfinite values cannot be plotted and are
excluded from chart statistics. Low finite scores are included.

Generate plots from existing evaluation CSVs without evaluating again:

```bash
python scripts/plot_results.py
```

## Paired model comparison and bootstrap intervals

After evaluating both models on the same BUSI test set:

```bash
python scripts/compare_models.py --config configs/comparison_busi.yaml
python scripts/bootstrap_comparison.py --config configs/comparison_busi.yaml
```

Paths, metrics, bootstrap resamples, seed, and output directory come from the
comparison YAML. The default paths match `runs/busi_pb/evaluation` and
`runs/busi_unet/evaluation`. For Sunnybrook, use
`--config configs/comparison_sunnybrook.yaml`. For postprocessed results, copy
the config and change both input CSV paths and the output directory.

The scripts pair rows by `image_path`, require identical unique image identities,
and reject different target mask paths when both CSVs include them. Row order
does not affect comparison. Set `pair_key` to another shared unique identifier
if evaluation paths differ across systems; the scripts do not guess matches.

`compare_models.py` saves `paired_comparison.csv` and
`paired_comparison.json`, including finite pair counts, excluded nonfinite pairs,
means, mean/median PB-minus-U-Net differences, wins/losses/ties, and two-sided
Wilcoxon statistics and p-values. HD95 wins mean a lower distance; other metrics
use a higher score. All-zero differences return p=1. Missing metrics cause an
error; remove HD95 from the YAML list if it was not evaluated. P-values are
unadjusted exploratory comparisons across metrics. Tie tolerance affects win
counts only; Wilcoxon uses the original differences.

`bootstrap_comparison.py` saves `bootstrap_dice.json` and
`bootstrap_dice_samples.npz` with the mean and median bootstrap distributions.
It uses 10,000 paired resamples, seed 42, and 95% percentile intervals by
default. Positive Dice differences favor PB-U-Net. Override
`--bootstrap-metric iou` to analyze another metric.

Both analyses use images as the sampling unit. If several images belong to the
same patient, these procedures do not account for within-patient dependence;
patient-level analysis requires patient identities and grouped resampling.
