"""Download original X-rays and both manual lung masks from the NLM host."""
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.request import urlopen

import cv2
from tqdm.auto import tqdm

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


def download_montgomery(config, dry_run=False):
    """Retrieve configured sources; dry runs resolve remote listings without writing files."""
    dataset = config["dataset"]
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
    report = {"dataset": "montgomery", "raw_root": str(root.resolve()),
              "images": len(names), "masks": 2 * len(names), "dry_run": dry_run}
    if dry_run:
        return report
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
    return report
