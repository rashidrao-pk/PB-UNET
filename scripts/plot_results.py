#!/usr/bin/env python3
"""Generate plots from existing results without rerunning training or evaluation."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.plotting import plot_training_history, plot_evaluation_metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs",
                        help="Scan this directory for saved history JSON and per-image metric CSVs")
    args = parser.parse_args()
    for path in sorted(args.runs_dir.rglob("history.json")):
        plot_training_history(json.loads(path.read_text()), path.parent)
        print(f"Training plots: {path.parent}")
    for path in sorted(args.runs_dir.rglob("per_image_*.csv")):
        suffix = path.stem.removeprefix("per_image_")
        plot_evaluation_metrics(pd.read_csv(path), path.parent, suffix)
        print(f"Evaluation plots: {path.parent} ({suffix})")

if __name__ == "__main__":
    main()
