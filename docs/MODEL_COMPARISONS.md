# Architecture comparison protocol

This repository contains a controlled comparison suite for binary medical-image segmentation.

## Implemented models

- `unet`: manuscript baseline U-Net.
- `portable_bridge_legacy`: exact paper-era Portable-Bridge U-Net architecture for reproducibility.
- `portable_bridge`: revised PB-U-Net-R with bilinear feature alignment and a final decoder refinement block.
- `apb_unet`: PB-U-Net-R plus lightweight channel gating before bridge skip fusion.
- `unetpp`: nested dense-skip U-Net++ comparison.
- `attention_unet`: attention-gated skip U-Net.
- `resunet`: residual U-Net.
- `resunetpp`: residual + squeeze/excitation + ASPP + attention-gated decoder comparison.
- `unet3plus`: full-scale multi-level feature-fusion comparison inspired by U-Net 3+.

The comparison implementations are intentionally self-contained and use the same encoder widths supplied with `--filters`. They should be described as the implementations used in this repository, not as exact reproductions of every training detail from the original method papers.

## Losses

`train.py` supports:

- `bce` — paper-era parity.
- `bce_dice` — recommended revised segmentation objective.
- `dice`.
- `focal`.

For an architecture comparison, use the **same loss for every model**. Do not compare a BCE-trained U-Net against a BCE+Dice-trained PB-U-Net and attribute the entire gain to the architecture.

## One-command benchmark

Kvasir-SEG:

```bash
python scripts/run_benchmark.py --config configs/benchmark_kvasir_epito.yaml
```

BUSI:

```bash
python scripts/run_benchmark.py --config configs/benchmark_busi_epito.yaml
```

The benchmark trains each model using the exact same explicit split CSVs and settings, evaluates `best_dice.pt`, optionally evaluates post-processing, and creates:

```text
leaderboard_raw.csv
paired_vs_unet.csv
dice_bootstrap_vs_unet.json
<model>/best_dice.pt
<model>/best_loss.pt
<model>/evaluation/per_image_raw.csv
<model>/evaluation/summary_raw.json
```

Use `--dry-run` to inspect all commands without training, `--skip-training` to rebuild evaluation/statistics from existing checkpoints, and `--skip-evaluation` if only training is desired.

## Recommended ablation sequence

To isolate what improves PB-U-Net:

1. `portable_bridge_legacy` + BCE — original architecture.
2. `portable_bridge` + BCE — tests bilinear alignment + final refinement only.
3. `portable_bridge` + BCE+Dice — tests the revised loss.
4. `apb_unet` + BCE+Dice — tests gated feature transfer.
5. Compare the selected PB variant against all external architectures with the same BCE+Dice objective.

This prevents architectural and objective changes from being conflated.
