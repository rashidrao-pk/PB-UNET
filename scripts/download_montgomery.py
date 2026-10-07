#!/usr/bin/env python3
"""Compatibility CLI; implementation lives in portable_bridge_unet.data.retrieve."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portable_bridge_unet.data.cli import dataset_main


def main(argv=None):
    return dataset_main("retrieve", "montgomery", "montgomery_pb.yaml", argv)


if __name__ == "__main__":
    raise SystemExit(main())
