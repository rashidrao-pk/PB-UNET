#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portable_bridge_unet.config import DEFAULT_CONFIG, load_config
from portable_bridge_unet.preparation import plan_samples, prepare_dataset

def main():
    parser = argparse.ArgumentParser(description="Prepare Sunnybrook LV cavity PNG image/mask pairs from config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        plan = plan_samples(config)
        print(json.dumps({"pairs": len(plan), "patients": len({p[0] for p in plan})}, indent=2))
    else:
        print(json.dumps(prepare_dataset(config), indent=2))

if __name__ == "__main__":
    main()
