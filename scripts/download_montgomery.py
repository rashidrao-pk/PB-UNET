#!/usr/bin/env python3
"""Download original X-rays and both manual lung masks from the NLM host."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.request import urlopen

import cv2
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.config import load_config

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/"


class FileLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if re.fullmatch(r"MCUCXR_\d+_[01]\.png", href):
                self.names.add(href)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/montgomery_pb.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Check remote file lists without downloading images")
    args = parser.parse_args()
    try:
        dataset = load_config(args.config)["dataset"]
        root = Path(dataset["raw_root"])
        inventories = {}
        for folder in ("CXR_png", "ManualMask/leftMask", "ManualMask/rightMask"):
            with urlopen(BASE + folder + "/index.html", timeout=60) as response:
                links = FileLinks()
                links.feed(response.read().decode("utf-8"))
            inventories[folder] = links.names
        names = inventories["CXR_png"]
        expected = dataset.get("expected_pairs", 138)
        if not names or any(files != names for files in inventories.values()):
            raise ValueError("Remote image and mask filename sets differ or are empty")
        if expected is not None and len(names) != expected:
            raise ValueError(f"Expected {expected} remote images, found {len(names)}")
        print(f"NLM source: {len(names)} X-rays and {2 * len(names)} masks; destination: {root}")
        if args.dry_run:
            return
        jobs = [(folder, name) for folder in inventories for name in sorted(names)]
        for folder, name in tqdm(jobs, desc="Download Montgomery", unit="file"):
            path = root / folder / name
            if path.is_file() and cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".part")
            with urlopen(BASE + folder + "/" + name, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if cv2.imread(str(temporary), cv2.IMREAD_UNCHANGED) is None:
                raise ValueError(f"Downloaded file is not a readable image: {name}")
            temporary.replace(path)
        print("Download complete. Run scripts/prepare_montgomery.py with the same config.")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Download failed: {exc}\nRe-run to retain completed files; see docs/MONTGOMERY.md.\n")


if __name__ == "__main__":
    main()
