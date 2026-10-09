#!/usr/bin/env python3
"""Preflight and run the configured datasets, including external evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portable_bridge_unet.experiment_suite import run_suite


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/benchmark_all_epito.yaml'))
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--mode',choices=['benchmark','multiseed'],default='benchmark')
    parser.add_argument('--out-dir')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--prepare',action='store_true',help='Prepare missing data/splits from existing raw downloads; no downloads')
    parser.add_argument('--skip-training',action='store_true')
    args=parser.parse_args()
    try:
        report=run_suite(args.config,selected=args.datasets,mode=args.mode,out_dir=args.out_dir,
                         dry_run=args.dry_run,check_only=args.check_only,prepare=args.prepare,skip_training=args.skip_training)
        print(json.dumps(report,indent=2))
        return 0 if report['status'] in ('planned','ready','complete') else 1
    except (OSError,ValueError,KeyError) as exc: parser.exit(1,f'Suite failed: {exc}\n')


if __name__=='__main__': raise SystemExit(main())
