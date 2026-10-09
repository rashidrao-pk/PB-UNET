#!/usr/bin/env python3
"""Generate the existing dataset smoke-test figures for several datasets."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [ROOT / "configs" / name for name in (
    "sunnybrook_pb_epito.yaml", "busi_pb_epito.yaml", "kvasir_pb_epito.yaml",
    "montgomery_pb_epito.yaml", "isic2016_pb_epito.yaml",
    "brats2020_pb_epito.yaml", "brats2021_pb_epito.yaml", "drive_pb_epito.yaml", "cvc_clinicdb_external_epito.yaml")]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=DEFAULT_CONFIGS,
                        help="Dataset configs (default: all nine configured Epito datasets)")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/smoke_test",
                        help="Parent directory; each dataset gets its own subdirectory")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    failed = []
    for path in args.configs:
        try:
            config = load_config(path)
            name = config["dataset"].get("name", path.stem)
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("dataset.name must be a single directory name")
            print(f"\n=== {name}: labeled dataset previews ===", flush=True)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/check_dataset.py"),
                "--config", str(path.resolve()), "--require-prepared",
                "--num-samples", str(args.num_samples),
                "--out-dir", str(args.out_dir.resolve() / name),
            ])
            if result.returncode:
                failed.append(str(path))
        except (OSError, ValueError, KeyError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            failed.append(str(path))
    if failed:
        print("\nSome dataset checks failed; inspect their reports: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"\nAll previews saved under {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
