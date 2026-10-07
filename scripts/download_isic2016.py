#!/usr/bin/env python3
"""Download the four ISIC 2016 Task 1 archives from official challenge links."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen
import zipfile

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portable_bridge_unet.config import load_config
from portable_bridge_unet.isic2016 import FOLDERS

PAGE = "https://challenge.isic-archive.com/data/"


class ArchiveLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {}

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        url = urljoin(PAGE, href)
        name = Path(unquote(urlsplit(url).path)).name
        if urlsplit(url).scheme == "https" and name in {folder + ".zip" for folder in FOLDERS.values()}:
            self.links[name] = url


def extract_images(archive, destination, masks=False):
    """Extract only expected image basenames into a known directory."""
    pattern = r"ISIC_\d+_segmentation\.png" if masks else r"ISIC_\d+\.(?:jpg|jpeg|png)"
    with zipfile.ZipFile(archive) as source:
        members, seen = [], set()
        for member in source.infolist():
            name = Path(member.filename).name
            if member.is_dir() or "__MACOSX" in Path(member.filename).parts:
                continue
            if not re.fullmatch(pattern, name, re.IGNORECASE):
                continue
            if name.lower() in seen:
                raise ValueError(f"Duplicate image filename in {archive}: {name}")
            seen.add(name.lower())
            members.append((member, name))
        if not members:
            raise ValueError(f"No expected segmentation {'masks' if masks else 'images'} in {archive}")
        destination.mkdir(parents=True, exist_ok=True)
        for member, name in tqdm(members, desc=f"Extract {destination.name}", unit="file"):
            target = destination / name
            temporary = target.with_suffix(target.suffix + ".part")
            with source.open(member) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isic2016_pb.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Resolve official links without downloading archives")
    args = parser.parse_args()
    try:
        dataset = load_config(args.config)["dataset"]
        with urlopen(PAGE, timeout=60) as response:
            links = ArchiveLinks()
            links.feed(response.read().decode("utf-8"))
        expected = {folder + ".zip" for folder in FOLDERS.values()}
        if expected - links.links.keys():
            raise ValueError("Official Task 1 archive links missing: " + ", ".join(sorted(expected - links.links.keys())))
        root = Path(dataset["raw_root"])
        for role, folder in FOLDERS.items():
            filename = folder + ".zip"
            url = links.links[filename]
            print(f"{role}: {url}", flush=True)
            if args.dry_run:
                continue
            archive = root / "archives" / filename
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not zipfile.is_zipfile(archive):
                temporary = archive.with_suffix(".zip.part")
                with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                    total = response.headers.get("Content-Length")
                    with tqdm(total=int(total) if total else None, desc=filename, unit="B", unit_scale=True) as progress:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                            progress.update(len(chunk))
                if not zipfile.is_zipfile(temporary):
                    raise ValueError(f"Not a valid ZIP download: {filename}")
                temporary.replace(archive)
            extract_images(archive, root / folder, masks=role.endswith("masks"))
        if not args.dry_run:
            print("Download complete. Run scripts/prepare_isic2016.py with the same config.")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Download failed: {exc}\nSee docs/ISIC2016.md for official/manual downloads.\n")


if __name__ == "__main__":
    main()
