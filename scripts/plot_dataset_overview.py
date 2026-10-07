#!/usr/bin/env python3
"""Combine saved dataset smoke previews and descriptions into one figure."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.data.overview import DATASETS, create_dataset_overview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, default=ROOT / "runs/smoke_test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/dataset_overview")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    try:
        outputs = create_dataset_overview(args.smoke_dir, args.out_dir, args.sample_index, args.datasets)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
