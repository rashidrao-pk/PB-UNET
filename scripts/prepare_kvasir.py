#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portable_bridge_unet.config import load_config
from portable_bridge_unet.kvasir import prepare_kvasir

def main():
    parser = argparse.ArgumentParser(description="Validate and prepare Kvasir-SEG RGB images and binary polyp masks")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/kvasir_pb.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_kvasir(load_config(args.config), args.dry_run), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\nDownload and extract Kvasir-SEG first; see docs/KVASIR.md.\n")

if __name__ == "__main__":
    main()
