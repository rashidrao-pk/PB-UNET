"""Paired per-image comparisons; CSV row order never determines pairing."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/comparison_busi.yaml"

def comparison_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    probe, _ = parser.parse_known_args()
    path = Path(probe.config).expanduser().resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        parser.error("comparison config must be a mapping")
    for key in ("pb_csv", "unet_csv", "out_dir"):
        if key in config:
            value = Path(config[key]).expanduser()
            config[key] = str(value if value.is_absolute() else path.parent / value)
        parser.add_argument("--" + key.replace("_", "-"))
    parser.add_argument("--pair-key", default="image_path")
    parser.add_argument("--metrics", nargs="+", default=["dice", "iou", "precision", "recall", "hd95"])
    parser.add_argument("--bootstrap-metric", default="dice")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tie-tolerance", type=float, default=1e-8)
    parser.set_defaults(**config)
    args = parser.parse_args()
    for key in ("pb_csv", "unet_csv", "out_dir"):
        if not getattr(args, key):
            parser.error(f"{key} must be supplied by config or CLI")
    if args.bootstrap_resamples < 1 or args.tie_tolerance < 0:
        parser.error("bootstrap resamples must be positive and tie tolerance nonnegative")
    return args

def read_paired(pb_path, unet_path, key="image_path"):
    pb, unet = pd.read_csv(pb_path), pd.read_csv(unet_path)
    for name, frame in (("PB", pb), ("U-Net", unet)):
        if key not in frame:
            raise ValueError(f"{name}: missing pairing column {key}")
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name}: pairing identities must be non-null and unique")
    if pb.empty:
        raise ValueError("no images to compare")
    if set(pb[key]) != set(unet[key]):
        raise ValueError("image identities differ between evaluation CSVs; evaluate the same test set")
    pb = pb.set_index(key).sort_index()
    unet = unet.set_index(key).reindex(pb.index)
    if "mask_path" in pb and "mask_path" in unet and not pb["mask_path"].equals(unet["mask_path"]):
        raise ValueError("paired images have different target mask paths")
    return pb, unet

def metric_arrays(pb, unet, metric):
    if metric not in pb or metric not in unet:
        raise ValueError(f"metric missing from an evaluation CSV: {metric}")
    a, b = pb[metric].to_numpy(float), unet[metric].to_numpy(float)
    valid = np.isfinite(a) & np.isfinite(b)
    return a[valid], b[valid], valid

def paired_statistics(pb, unet, metrics, tolerance=1e-8):
    rows = []
    for metric in metrics:
        a, b, valid = metric_arrays(pb, unet, metric)
        delta = a - b
        advantage = -delta if metric == "hd95" else delta
        n = len(delta)
        if not n:
            statistic, p, status = None, None, "no finite pairs"
        elif np.all(delta == 0):
            statistic, p, status = 0., 1., "all paired differences are zero"
        else:
            result = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto")
            statistic, p, status = float(result.statistic), float(result.pvalue), "ok"
        rows.append(dict(metric=metric, n=n, excluded_nonfinite=int((~valid).sum()),
                         pb_mean=float(a.mean()) if n else None,
                         unet_mean=float(b.mean()) if n else None,
                         mean_delta=float(delta.mean()) if n else None,
                         median_delta=float(np.median(delta)) if n else None,
                         pb_wins=int((advantage > tolerance).sum()),
                         unet_wins=int((advantage < -tolerance).sum()),
                         ties=int((np.abs(advantage) <= tolerance).sum()),
                         lower_is_better=metric == "hd95",
                         wilcoxon_statistic=statistic, wilcoxon_p=p, status=status))
    return rows

def bootstrap_differences(pb, unet, metric="dice", resamples=10000, seed=42):
    if resamples < 1:
        raise ValueError("resamples must be positive")
    a, b, valid = metric_arrays(pb, unet, metric)
    delta = a - b
    if not len(delta):
        raise ValueError(f"{metric}: no finite pairs for bootstrap")
    rng = np.random.default_rng(seed)
    means, medians = np.empty(resamples), np.empty(resamples)
    for i in range(resamples):
        sample = rng.choice(delta, size=len(delta), replace=True)
        means[i], medians[i] = sample.mean(), np.median(sample)
    return dict(metric=metric, n=len(delta), excluded_nonfinite=int((~valid).sum()),
                delta_definition="PB minus U-Net", seed=seed, resamples=resamples,
                confidence_level=.95, method="paired per-image percentile bootstrap",
                mean_delta=float(delta.mean()), mean_ci95=np.percentile(means, [2.5, 97.5]).tolist(),
                median_delta=float(np.median(delta)), median_ci95=np.percentile(medians, [2.5, 97.5]).tolist()), means, medians
