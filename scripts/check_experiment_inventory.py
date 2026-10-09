#!/usr/bin/env python3
"""Report settings, split availability and completed evaluation artifacts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from portable_bridge_unet.experiment_suite import load_suite
from portable_bridge_unet.utils import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/benchmark_all_epito.yaml'))
    parser.add_argument('--runs-dir',type=Path,default=Path(__file__).resolve().parents[1]/'runs')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    _,entries=load_suite(args.config)
    report={}
    for row in entries:
        cfg=row['resolved_benchmark']
        splits=cfg['splits'] if row['role']=='train' else {'test':cfg['split_csv']}
        root=Path(cfg['out_root'])
        expected=[]
        for model in cfg['models']:
            expected.append(root/model/('evaluation/summary_raw.json' if row['role']=='train' else 'summary_raw.json'))
        suite_summaries=list(args.runs_dir.glob(f'dataset_suite/*/{row["name"]}/**/summary_raw.json'))
        report[row['name']]=dict(role=row['role'],experiment=row['experiment'],benchmark=row['benchmark'],
            multiseed=row.get('multiseed'),models=cfg['models'],
            splits={name:dict(path=path,available=Path(path).is_file()) for name,path in splits.items()},
            standard_run_evaluations_found=sum(path.is_file() for path in expected),
            suite_evaluations_found=len(suite_summaries),
            completed_results_verified=False)
    if args.out: save_json(report,args.out)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
