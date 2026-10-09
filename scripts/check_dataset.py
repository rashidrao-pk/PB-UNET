#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portable_bridge_unet.config import DEFAULT_CONFIG, load_config
from portable_bridge_unet.data.checks import check_dataset
from portable_bridge_unet.data.visualization import save_smoke_test

def main():
    parser = argparse.ArgumentParser(description="Check raw downloads and decode all prepared image/mask pairs")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--out-dir", help="Smoke artifact directory (default: config smoke_test.out_dir)")
    parser.add_argument("--num-samples", type=int, help="Number of image/mask previews")
    args = parser.parse_args()
    config = load_config(args.config)
    report = check_dataset(config)
    settings = config.get("smoke_test", {})
    out_dir = args.out_dir or settings.get("out_dir", str(Path(__file__).resolve().parents[1] / "runs/smoke_test"))
    num_samples = args.num_samples if args.num_samples is not None else settings.get("num_samples", 6)
    if num_samples < 1:
        parser.error("--num-samples must be positive")
    artifacts, smoke = save_smoke_test(config, report, out_dir, num_samples)
    report["smoke_artifacts"] = str(artifacts)
    report["pretraining_ready"] = smoke["pretraining_ready"]
    print(json.dumps(report, indent=2))
    return 0 if report["pretraining_ready" if args.require_prepared else "raw_available"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
