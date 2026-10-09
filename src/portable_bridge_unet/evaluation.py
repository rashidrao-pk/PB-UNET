from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .metrics import batch_metrics_from_logits, hd95_binary, summarize
from .postprocess import postprocess


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    apply_postprocessing: bool = False,
    compute_hd95: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    model.eval()
    rows: list[dict[str, object]] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        logits = model(images)

        if not apply_postprocessing:
            per_batch = batch_metrics_from_logits(logits, targets, threshold)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            for i in range(images.shape[0]):
                row: dict[str, object] = {
                    "image_path": batch["image_path"][i],
                    "mask_path": batch["mask_path"][i],
                }
                for k, vals in per_batch.items():
                    row[k] = float(vals[i].cpu())
                if compute_hd95:
                    row["hd95"] = hd95_binary(
                        preds[i, 0].cpu().numpy(), targets[i, 0].cpu().numpy()
                    )
                rows.append(row)
        else:
            probs = torch.sigmoid(logits).cpu().numpy()
            gt = targets.cpu().numpy()
            for i in range(images.shape[0]):
                raw = (probs[i, 0] >= threshold).astype(np.uint8) * 255
                pp = (postprocess(raw) > 0).astype(np.float32)
                pred_t = torch.from_numpy(pp)[None, None]
                target_t = torch.from_numpy(gt[i, 0])[None, None]
                # Convert postprocessed binary prediction into confident logits.
                pseudo_logits = torch.where(pred_t > 0.5, torch.tensor(20.0), torch.tensor(-20.0))
                vals = batch_metrics_from_logits(pseudo_logits, target_t, threshold=0.5)
                row = {
                    "image_path": batch["image_path"][i],
                    "mask_path": batch["mask_path"][i],
                }
                for k, v in vals.items():
                    row[k] = float(v[0])
                if compute_hd95:
                    row["hd95"] = hd95_binary(pp, gt[i, 0])
                rows.append(row)

    df = pd.DataFrame(rows)
    summary = {
        col: summarize(df[col].dropna().astype(float).tolist())
        for col in ["dice", "iou", "precision", "recall", "accuracy", "hd95"]
        if col in df.columns
    }
    return df, summary
