#!/usr/bin/env python3
"""Validate and prepare Montgomery X-rays and union-of-lungs masks."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.config import load_config
from portable_bridge_unet.montgomery import prepare_montgomery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/montgomery_pb.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_montgomery(load_config(args.config), args.dry_run), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\nSee docs/MONTGOMERY.md for download and path configuration.\n")


if __name__ == "__main__":
    main()
