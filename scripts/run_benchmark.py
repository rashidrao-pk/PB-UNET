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
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.comparison import read_paired, paired_statistics, bootstrap_differences
from portable_bridge_unet.models import MODEL_NAMES
from portable_bridge_unet.experiments import read_benchmark, validate_splits, training_arguments


def run(cmd: list[str], dry_run: bool = False) -> None:
    # Suspend the outer bar while subprocesses print their own epoch/batch bars.
    with tqdm.external_write_mode():
        print("\n$ " + " ".join(cmd), flush=True)
        if not dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark PB-U-Net against U-Net family baselines")
    p.add_argument("--config", required=True)
    p.add_argument("--out-root", help="Override the output root for a new experiment")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-evaluation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = read_benchmark(cfg_path)
    if args.out_root: cfg["out_root"] = str(Path(args.out_root).expanduser().resolve())
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
    common = training_arguments(cfg)
    if not args.dry_run and not (args.skip_training and args.skip_evaluation):
        validate_splits(cfg, require_groups=cfg.get('split_unit') == 'group')

    stages_per_model = int(not args.skip_training)
    if not args.skip_evaluation:
        stages_per_model += 1 + int(bool(cfg.get("postprocess", True)))
    with tqdm(total=len(models) * stages_per_model, desc="Benchmark", unit="stage",
              dynamic_ncols=True, disable=args.dry_run or stages_per_model == 0) as progress:
        for model in models:
            model_dir = out_root / model
            if not args.skip_training:
                progress.set_postfix(model=model, stage="training", refresh=True)
                run([sys.executable, "scripts/train.py", "--model", model, *common, "--out-dir", str(model_dir)], args.dry_run)
                progress.update(1)
            if not args.skip_evaluation:
                progress.set_postfix(model=model, stage="evaluation/raw", refresh=True)
                ckpt = model_dir / "best_dice.pt"
                eval_dir = model_dir / "evaluation"
                run([
                    sys.executable, "scripts/evaluate.py",
                    "--checkpoint", str(ckpt), "--split-csv", test_split,
                    "--out-dir", str(eval_dir), "--device", str(t.get("device", "auto")),
                    "--num-workers", str(t.get("num_workers", 4)),
                ], args.dry_run)
                progress.update(1)
                if cfg.get("postprocess", True):
                    progress.set_postfix(model=model, stage="evaluation/postprocessed", refresh=True)
                    run([
                        sys.executable, "scripts/evaluate.py",
                        "--checkpoint", str(ckpt), "--split-csv", test_split,
                        "--postprocess", "--out-dir", str(model_dir / "evaluation_postprocessed"),
                        "--device", str(t.get("device", "auto")),
                        "--num-workers", str(t.get("num_workers", 4)),
                    ], args.dry_run)

                    progress.update(1)

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
    if len(rows) != len(models):
        raise ValueError("Missing model evaluation summaries; refusing a partial leaderboard")
    leaderboard = pd.DataFrame(rows).sort_values("dice_mean", ascending=False)
    leaderboard.to_csv(out_root / "leaderboard_raw.csv", index=False)
    print("\n=== RAW LEADERBOARD ===")
    print(leaderboard.to_string(index=False))

    # Paired comparisons versus U-Net.
    baseline_csv = out_root / "unet" / "evaluation" / "per_image_raw.csv"
    comparison_rows = []
    bootstrap_reports = {}
    if baseline_csv.exists():
        for model in tqdm([name for name in models if name != "unet"],
                          desc="Paired statistics + bootstrap", unit="model", dynamic_ncols=True):
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
