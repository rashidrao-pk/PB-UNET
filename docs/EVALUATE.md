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
