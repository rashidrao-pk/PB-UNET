#!/usr/bin/env python3
"""Prepare ISIC 2016 Task 1 while preserving the official test set."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.config import load_config
from portable_bridge_unet.isic2016 import prepare_isic2016


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isic2016_pb.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_isic2016(load_config(args.config), args.dry_run), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\nSee docs/ISIC2016.md.\n")


if __name__ == "__main__":
    main()
