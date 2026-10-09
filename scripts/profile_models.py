#!/usr/bin/env python3
"""Report params, approximate Conv/Linear FLOPs, latency, throughput, and peak CUDA memory."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from portable_bridge_unet.models import MODEL_NAMES, build_model, count_trainable_parameters
from portable_bridge_unet.utils import resolve_device


def approx_flops(model, x):
    flops=0
    hooks=[]
    def conv_hook(m, inp, out):
        nonlocal flops
        # multiply + add = 2 FLOPs
        b, cout, h, w = out.shape
        cin=m.in_channels
        kh,kw=m.kernel_size
        groups=m.groups
        flops += int(b*h*w*cout*(cin//groups)*kh*kw*2)
        if m.bias is not None: flops += int(b*h*w*cout)
    def linear_hook(m, inp, out):
        nonlocal flops
        b=inp[0].shape[0] if inp[0].ndim>1 else 1
        flops += int(b*m.in_features*m.out_features*2)
    for m in model.modules():
        if isinstance(m,nn.Conv2d): hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m,nn.Linear): hooks.append(m.register_forward_hook(linear_hook))
    with torch.no_grad(): model(x)
    for h in hooks: h.remove()
    return flops


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--filters", type=int, nargs="+", default=[32,64,128])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/profiling/model_profile.csv")
    args=p.parse_args()
    device=resolve_device(args.device)
    rows=[]
    for name in args.models:
        model=build_model(name,args.channels,1,args.filters).to(device).eval()
        x=torch.randn(args.batch_size,args.channels,args.image_size,args.image_size,device=device)
        params=count_trainable_parameters(model)
        flops=approx_flops(model,x[:1])
        with torch.no_grad():
            for _ in range(args.warmup): model(x)
        if device.type=="cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        with torch.no_grad():
            for _ in range(args.iters): model(x)
        if device.type=="cuda": torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        latency_ms=elapsed/args.iters*1000
        fps=(args.batch_size*args.iters)/elapsed
        peak_mb=(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=="cuda" else float("nan")
        rows.append({"model":name,"params":params,"params_m":params/1e6,"flops":flops,"gflops":flops/1e9,"batch_size":args.batch_size,"latency_ms_per_batch":latency_ms,"images_per_second":fps,"peak_cuda_memory_mb":peak_mb,"device":str(device)})
        print(rows[-1])
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"Saved: {out}")

if __name__=="__main__": main()
