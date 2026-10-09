#!/usr/bin/env python3
"""Write patient-separated split CSVs from the Sunnybrook preparation manifest."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portable_bridge_unet.config import load_config
from portable_bridge_unet.data.splits import prepare_grouped_splits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1]/'configs/sunnybrook_pb.yaml'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        config=load_config(args.config)
        if config['dataset']['name']!='sunnybrook':
            raise ValueError('This script requires a Sunnybrook config')
        print(json.dumps(prepare_grouped_splits(config,args.dry_run),indent=2))
    except (OSError,ValueError) as exc:
        parser.exit(1,f'Split preparation failed: {exc}\n')


if __name__=='__main__': main()
