#!/usr/bin/env python3
"""Run repeated-seed training/evaluation for selected models on one fixed split."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, dry=False):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    args = p.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    resolve = lambda x: str(Path(x).expanduser() if Path(x).expanduser().is_absolute() else (cfg_path.parent / x).resolve())
    splits = cfg["splits"]
    tr, va, te = [resolve(splits[k]) for k in ("train", "val", "test")]
    models = cfg.get("models", ["unet", "portable_bridge_legacy"])
    seeds = [int(x) for x in cfg.get("seeds", [42, 123, 2026])]
    out_root = Path(resolve(cfg.get("out_root", "../runs/multiseed_kvasir")))
    t = cfg.get("training", {})
    common = [
        "--train-split", tr, "--val-split", va, "--test-split", te,
        "--epochs", str(t.get("epochs",100)), "--batch-size", str(t.get("batch_size",8)),
        "--image-size", str(t.get("image_size",256)), "--channels", str(t.get("channels",3)),
        "--filters", *map(str,t.get("filters",[32,64,128])), "--lr", str(t.get("lr",1e-4)),
        "--loss", str(t.get("loss","bce_dice")), "--num-workers", str(t.get("num_workers",4)),
        "--device", str(t.get("device","auto")), "--patience", str(t.get("patience",30)),
        "--lr-patience", str(t.get("lr_patience",10)), "--lr-factor", str(t.get("lr_factor",0.05)),
        "--threshold", str(t.get("threshold",0.5)),
    ]
    if t.get("augment",False): common.append("--augment")
    if not t.get("amp",True): common.append("--no-amp")

    for model in models:
        for seed in seeds:
            run_dir = out_root / model / f"seed_{seed}"
            if not args.skip_training:
                run([sys.executable,"scripts/train.py","--model",model,"--seed",str(seed),*common,"--out-dir",str(run_dir)], args.dry_run)
            run([sys.executable,"scripts/evaluate.py","--checkpoint",str(run_dir/"best_dice.pt"),"--split-csv",te,"--out-dir",str(run_dir/"evaluation"),"--device",str(t.get("device","auto"))], args.dry_run)

    if args.dry_run: return
    rows=[]
    for model in models:
        for seed in seeds:
            pth=out_root/model/f"seed_{seed}"/"evaluation"/"summary_raw.json"
            if not pth.exists(): continue
            s=json.loads(pth.read_text())
            rows.append({"model":model,"seed":seed,**{m:s[m]["mean"] for m in ("dice","iou","precision","recall","accuracy","hd95")}})
    raw=pd.DataFrame(rows)
    raw.to_csv(out_root/"per_seed_metrics.csv",index=False)
    agg=raw.groupby("model").agg({m:["mean","std"] for m in ("dice","iou","precision","recall","accuracy","hd95")})
    agg.columns=[f"{a}_{b}" for a,b in agg.columns]
    agg.reset_index().to_csv(out_root/"multiseed_summary.csv",index=False)
    print("\n=== MULTI-SEED SUMMARY ===")
    print(agg.to_string())

if __name__ == "__main__": main()
