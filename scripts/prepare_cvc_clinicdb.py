#!/usr/bin/env python3
"""Prepare external CVC-ClinicDB data from YAML or legacy explicit path options."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portable_bridge_unet.config import load_config
from portable_bridge_unet.data.preprocess.cvc import prepare_cvc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/cvc_clinicdb_external.yaml'))
    parser.add_argument('--raw-root')
    parser.add_argument('--prepared-root')
    parser.add_argument('--mask-threshold',type=int,default=127)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        config=load_config(args.config)
        for key in ('raw_root','prepared_root'):
            value=getattr(args,key)
            if value: config['dataset'][key]=str(Path(value).expanduser().resolve())
        if args.mask_threshold!=127: raise ValueError('CVC binary 0/255 annotations use threshold 127')
        print(json.dumps(prepare_cvc(config,args.dry_run),indent=2))
    except (OSError,ValueError) as exc: parser.exit(1,f'CVC preparation failed: {exc}\n')


if __name__=='__main__': main()
