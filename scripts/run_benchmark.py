#!/usr/bin/env python3
"""Train/evaluate all segmentation architectures under one identical protocol.

Example:
  python scripts/run_benchmark.py --config configs/benchmark_kvasir_epito.yaml

The script intentionally reuses one explicit train/val/test split for every model.
It produces raw and optional postprocessed evaluation, a leaderboard, and paired
Wilcoxon + Dice bootstrap comparisons against U-Net.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.comparison import read_paired, paired_statistics, bootstrap_differences
from portable_bridge_unet.models import MODEL_NAMES


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark PB-U-Net against U-Net family baselines")
    p.add_argument("--config", required=True)
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("benchmark config must be a mapping")

    def resolve(value: str) -> str:
        pth = Path(value).expanduser()
        return str(pth if pth.is_absolute() else (cfg_path.parent / pth).resolve())

    models = cfg.get("models", list(MODEL_NAMES))
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown models: {unknown}; available={MODEL_NAMES}")

    split = cfg["splits"]
    train_split, val_split, test_split = map(resolve, (split["train"], split["val"], split["test"]))
    out_root = Path(resolve(cfg.get("out_root", "../runs/benchmark")))
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    t = cfg.get("training", {})
    common = [
        "--seed", str(t.get("seed", cfg.get("seed", 42))),
        "--train-split", train_split,
        "--val-split", val_split,
        "--test-split", test_split,
        "--epochs", str(t.get("epochs", 100)),
        "--batch-size", str(t.get("batch_size", 8)),
        "--image-size", str(t.get("image_size", 256)),
        "--channels", str(t.get("channels", 3)),
        "--filters", *map(str, t.get("filters", [32, 64, 128])),
        "--lr", str(t.get("lr", 1e-4)),
        "--loss", str(t.get("loss", "bce_dice")),
        "--num-workers", str(t.get("num_workers", 4)),
        "--device", str(t.get("device", "auto")),
        "--patience", str(t.get("patience", 30)),
        "--lr-patience", str(t.get("lr_patience", 10)),
        "--lr-factor", str(t.get("lr_factor", 0.05)),
        "--threshold", str(t.get("threshold", 0.5)),
    ]
    common.append("--augment" if t.get("augment", False) else "--no-augment")
    if not t.get("amp", True):
        common.append("--no-amp")

    for model in models:
        model_dir = out_root / model
        if not args.skip_training:
            run([sys.executable, "scripts/train.py", "--model", model, *common, "--out-dir", str(model_dir)], args.dry_run)
        if not args.skip_evaluation:
            ckpt = model_dir / "best_dice.pt"
            eval_dir = model_dir / "evaluation"
            run([
                sys.executable, "scripts/evaluate.py",
                "--checkpoint", str(ckpt), "--split-csv", test_split,
                "--out-dir", str(eval_dir), "--device", str(t.get("device", "auto")),
                "--num-workers", str(t.get("num_workers", 4)),
            ], args.dry_run)
            if cfg.get("postprocess", True):
                run([
                    sys.executable, "scripts/evaluate.py",
                    "--checkpoint", str(ckpt), "--split-csv", test_split,
                    "--postprocess", "--out-dir", str(model_dir / "evaluation_postprocessed"),
                    "--device", str(t.get("device", "auto")),
                    "--num-workers", str(t.get("num_workers", 4)),
                ], args.dry_run)

    if args.dry_run or (args.skip_evaluation and not args.skip_training):
        return

    # Leaderboard from raw evaluation.
    rows = []
    for model in models:
        summary_path = out_root / model / "evaluation" / "summary_raw.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        rows.append({
            "model": model,
            **{f"{m}_mean": summary[m]["mean"] for m in ("dice", "iou", "precision", "recall", "accuracy", "hd95")},
            **{f"{m}_median": summary[m]["median"] for m in ("dice", "iou", "precision", "recall", "hd95")},
        })
    if not rows:
        raise ValueError("No evaluation summaries found; run evaluation before building the leaderboard")
    leaderboard = pd.DataFrame(rows).sort_values("dice_mean", ascending=False)
    leaderboard.to_csv(out_root / "leaderboard_raw.csv", index=False)
    print("\n=== RAW LEADERBOARD ===")
    print(leaderboard.to_string(index=False))

    # Paired comparisons versus U-Net.
    baseline_csv = out_root / "unet" / "evaluation" / "per_image_raw.csv"
    comparison_rows = []
    bootstrap_reports = {}
    if baseline_csv.exists():
        for model in models:
            if model == "unet":
                continue
            model_csv = out_root / model / "evaluation" / "per_image_raw.csv"
            if not model_csv.exists():
                continue
            candidate, baseline = read_paired(model_csv, baseline_csv, key="image_path")
            stats = paired_statistics(candidate, baseline, ["dice", "iou", "precision", "recall", "hd95"])
            for row in stats:
                row["model"] = model
                comparison_rows.append(row)
            boot, _, _ = bootstrap_differences(candidate, baseline, metric="dice", resamples=int(cfg.get("bootstrap_resamples", 10000)), seed=int(cfg.get("seed", 42)))
            boot["delta_definition"] = f"{model} minus U-Net"
            bootstrap_reports[model] = boot

    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(out_root / "paired_vs_unet.csv", index=False)
    (out_root / "dice_bootstrap_vs_unet.json").write_text(json.dumps(bootstrap_reports, indent=2, allow_nan=False))
    print(f"\nSaved benchmark outputs to: {out_root}")


if __name__ == "__main__":
    main()
