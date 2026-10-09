#!/usr/bin/env python3
"""Evaluate source-trained checkpoints on an external dataset without fine-tuning."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import yaml
from tqdm.auto import tqdm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from portable_bridge_unet.models import MODEL_NAMES
from portable_bridge_unet.data import load_samples_csv


def read_external(path):
    path=Path(path).resolve()
    config=yaml.safe_load(path.read_text())
    if not isinstance(config,dict): raise ValueError('External config must be a mapping')
    for key in ('split_csv','source_root','out_root'):
        p=Path(config[key]).expanduser()
        config[key]=str(p.resolve() if p.is_absolute() else (path.parent/p).resolve())
    models=config.get('models',list(MODEL_NAMES))
    if not models or len(set(models))!=len(models) or set(models)-set(MODEL_NAMES): raise ValueError('Invalid external model list')
    config['models']=models
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--source-root')
    parser.add_argument('--out-root')
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    cfg=read_external(args.config)
    for key in ('source_root','out_root'):
        if getattr(args,key): cfg[key]=str(Path(getattr(args,key)).resolve())
    seeds=args.seeds or cfg.get('seeds',[None])
    if len(set(seeds))!=len(seeds): raise ValueError('Duplicate external seeds')
    plan=[]
    for model in cfg['models']:
        for seed in seeds:
            suffix=Path(model) if seed is None else Path(model)/f'seed_{seed}'
            ckpt=Path(cfg['source_root'])/suffix/'best_dice.pt'
            out=Path(cfg['out_root'])/suffix
            plan.append((model,seed,ckpt,out))
    if not args.dry_run:
        load_samples_csv(cfg['split_csv'])
        missing=[str(item[2]) for item in plan if not item[2].is_file()]
        if missing: raise FileNotFoundError(f'Missing source checkpoints: {missing}')
    rows=[]
    for model,seed,checkpoint,out in tqdm(plan,desc='External evaluation',disable=args.dry_run):
        command=[sys.executable,str(ROOT/'scripts/evaluate.py'),'--checkpoint',str(checkpoint),'--split-csv',cfg['split_csv'],
                 '--out-dir',str(out),'--device',cfg.get('device','auto'),'--num-workers',str(cfg.get('num_workers',4))]
        print('$ '+' '.join(command),flush=True)
        if args.dry_run: continue
        subprocess.run(command,cwd=ROOT,check=True)
        summary=json.loads((out/'summary_raw.json').read_text())
        rows.append(dict(model=model,seed=seed,**{key:value['mean'] for key,value in summary.items()}))
    if rows:
        root=Path(cfg['out_root']);root.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(root/'external_metrics.csv',index=False)
        (root/'external_protocol.json').write_text(json.dumps({**cfg,'seeds':seeds,'fine_tuning':False},indent=2))


if __name__=='__main__': main()
