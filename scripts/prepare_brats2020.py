#!/usr/bin/env python3
"""Config-driven preparation adapter; algorithms live in the library."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portable_bridge_unet.data.cli import dataset_main


def main(argv=None):
    return dataset_main("preprocess", "brats2020", "brats2020_pb.yaml", argv)


if __name__ == "__main__":
    raise SystemExit(main())
