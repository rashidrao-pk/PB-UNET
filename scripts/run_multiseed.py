#!/usr/bin/env python3
"""Run repeated-seed training/evaluation for selected models on one fixed split."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from portable_bridge_unet.experiments import read_benchmark, validate_splits, training_arguments
from tqdm.auto import tqdm


def run(cmd, dry=False):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out-root", help="Override the output root for a new experiment")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    args = p.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = read_benchmark(cfg_path)
    if args.out_root: cfg["out_root"] = str(Path(args.out_root).expanduser().resolve())
    resolve = lambda x: str(Path(x).expanduser() if Path(x).expanduser().is_absolute() else (cfg_path.parent / x).resolve())
    splits = cfg["splits"]
    tr, va, te = [resolve(splits[k]) for k in ("train", "val", "test")]
    models = cfg.get("models", ["unet", "portable_bridge_legacy"])
    seeds = cfg.get("seeds", [42, 123, 2026])
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in seeds):
        raise ValueError("seeds must be nonempty unique nonnegative integers")
    out_root = Path(resolve(cfg.get("out_root", "../runs/multiseed_kvasir")))
    t = cfg.get("training", {})
    if not args.dry_run:
        validate_splits(cfg, require_groups=cfg.get('split_unit') == 'group')
        out_root.mkdir(parents=True,exist_ok=True)

    for model in tqdm(models,desc="Multi-seed models",disable=args.dry_run):
        for seed in tqdm(seeds,desc=model,leave=False,disable=args.dry_run):
            run_dir = out_root / model / f"seed_{seed}"
            if not args.skip_training:
                run([sys.executable,"scripts/train.py","--model",model,*training_arguments(cfg,seed),"--out-dir",str(run_dir)], args.dry_run)
            run([sys.executable,"scripts/evaluate.py","--checkpoint",str(run_dir/"best_dice.pt"),"--split-csv",te,"--out-dir",str(run_dir/"evaluation"),"--device",str(t.get("device","auto")),"--num-workers",str(t.get("num_workers",4))], args.dry_run)

    if args.dry_run: return
    rows=[]
    for model in models:
        for seed in seeds:
            pth=out_root/model/f"seed_{seed}"/"evaluation"/"summary_raw.json"
            if not pth.exists(): continue
            s=json.loads(pth.read_text())
            rows.append({"model":model,"seed":seed,**{m:s[m]["mean"] for m in ("dice","iou","precision","recall","accuracy","hd95")}})
    if len(rows) != len(models)*len(seeds):
        raise ValueError("Incomplete multi-seed evaluation outputs; refusing a partial summary")
    raw=pd.DataFrame(rows)
    raw.to_csv(out_root/"per_seed_metrics.csv",index=False)
    agg=raw.groupby("model").agg({m:["mean","std"] for m in ("dice","iou","precision","recall","accuracy","hd95")})
    agg.columns=[f"{a}_{b}" for a,b in agg.columns]
    agg.reset_index().to_csv(out_root/"multiseed_summary.csv",index=False)
    print("\n=== MULTI-SEED SUMMARY ===")
    print(agg.to_string())

if __name__ == "__main__": main()
