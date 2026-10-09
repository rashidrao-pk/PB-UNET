#!/usr/bin/env python3
"""Select and render PB-vs-U-Net wins, comparable cases, and PB failures."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from portable_bridge_unet.checkpoints import load_checkpoint
from portable_bridge_unet.utils import resolve_device


def preprocess(path,size,channels):
    flag=cv2.IMREAD_GRAYSCALE if channels==1 else cv2.IMREAD_COLOR
    x=cv2.imread(path,flag)
    if x is None: raise FileNotFoundError(path)
    if channels==3: x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    orig=x.copy(); x=cv2.resize(x,(size,size),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    if channels==1: t=torch.from_numpy(x[None,None])
    else: t=torch.from_numpy(np.transpose(x,(2,0,1))[None])
    return orig,t


def predict(model,ckpt,t,device):
    with torch.no_grad(): p=torch.sigmoid(model(t.to(device)))[0,0].cpu().numpy()
    return p >= float(ckpt.get("threshold",0.5))


def overlay(img, mask):
    if img.ndim==2: img=np.repeat(img[...,None],3,axis=2)
    out=img.astype(np.float32).copy()
    if out.max()<=1: out*=255
    # no explicit color styling; use grayscale bright overlay for reproducibility
    out[mask]=0.55*out[mask]+0.45*255
    return np.clip(out,0,255).astype(np.uint8)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pb-csv", required=True); p.add_argument("--unet-csv", required=True)
    p.add_argument("--pb-checkpoint", required=True); p.add_argument("--unet-checkpoint", required=True)
    p.add_argument("--n", type=int, default=5); p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="runs/qualitative_kvasir")
    args=p.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    pb=pd.read_csv(args.pb_csv); un=pd.read_csv(args.unet_csv)
    df=pb.merge(un,on=["image_path","mask_path"],suffixes=("_pb","_unet"))
    df["delta_dice"]=df.dice_pb-df.dice_unet
    n=args.n
    selected=pd.concat([
        df.nlargest(n,"delta_dice").assign(category="PB wins"),
        df.iloc[np.argsort(np.abs(df.delta_dice.to_numpy()))[:n]].assign(category="Comparable"),
        df.nsmallest(n,"delta_dice").assign(category="PB failures"),
    ],ignore_index=True)
    selected.to_csv(out/"selected_cases.csv",index=False)
    device=resolve_device(args.device); pbm,pbc=load_checkpoint(args.pb_checkpoint,device); unm,unc=load_checkpoint(args.unet_checkpoint,device)
    size=int(pbc.get("image_size",256)); channels=int(pbc.get("in_channels",3))
    for category,g in selected.groupby("category",sort=False):
        rows=[]
        fig,axes=plt.subplots(len(g),5,figsize=(16,3.2*len(g)),squeeze=False)
        for r,(_,row) in enumerate(g.iterrows()):
            img,t=preprocess(row.image_path,size,channels)
            gt=cv2.imread(row.mask_path,cv2.IMREAD_GRAYSCALE)>127
            gt=cv2.resize(gt.astype(np.uint8),(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)>0
            pp=predict(pbm,pbc,t,device); up=predict(unm,unc,t,device)
            pp=cv2.resize(pp.astype(np.uint8),(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)>0
            up=cv2.resize(up.astype(np.uint8),(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)>0
            panels=[img,gt,up,pp,overlay(img,pp)]
            titles=["Image","Ground truth",f"U-Net D={row.dice_unet:.3f}",f"PB D={row.dice_pb:.3f}",f"PB overlay Δ={row.delta_dice:+.3f}"]
            for c,(panel,title) in enumerate(zip(panels,titles)):
                axes[r,c].imshow(panel,cmap="gray" if panel.ndim==2 else None); axes[r,c].set_title(title); axes[r,c].axis("off")
        fig.tight_layout(); fname=category.lower().replace(" ","_")+".png"; fig.savefig(out/fname,dpi=200,bbox_inches="tight"); plt.close(fig)
    print(f"Saved qualitative panels and selected_cases.csv to {out}")

if __name__=="__main__": main()
