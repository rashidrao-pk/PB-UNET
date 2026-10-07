#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from portable_bridge_unet.comparison import comparison_args, read_paired, paired_statistics

def main():
    args = comparison_args("Paired PB-U-Net versus U-Net metrics and two-sided Wilcoxon tests")
    pb, unet = read_paired(args.pb_csv, args.unet_csv, args.pair_key)
    rows = paired_statistics(pb, unet, args.metrics, args.tie_tolerance)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "paired_comparison.csv", index=False)
    report = dict(settings=vars(args), paired_images=len(pb), metrics=rows,
                  delta_definition="PB minus U-Net",
                  p_values="Unadjusted two-sided Wilcoxon; exploratory per-metric tests",
                  sampling_unit="image; repeated images from a patient are not independent")
    (out / "paired_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
