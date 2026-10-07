#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portable_bridge_unet.comparison import comparison_args, read_paired, bootstrap_differences

def main():
    args = comparison_args("Paired bootstrap 95% confidence intervals for PB minus U-Net")
    pb, unet = read_paired(args.pb_csv, args.unet_csv, args.pair_key)
    report, means, medians = bootstrap_differences(
        pb, unet, args.bootstrap_metric, args.bootstrap_resamples, args.seed)
    report["settings"] = vars(args)
    report["sampling_unit"] = "image; not a patient-level bootstrap"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metric = args.bootstrap_metric
    (out / f"bootstrap_{metric}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(out / f"bootstrap_{metric}_samples.npz", mean=means, median=medians)
    print(json.dumps(report, indent=2))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
