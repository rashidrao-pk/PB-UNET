"""Download the four ISIC 2016 Task 1 archives from official challenge links."""
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen
import zipfile

from tqdm.auto import tqdm

from ..preprocess.isic2016 import FOLDERS

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


def download_isic2016(config, dry_run=False):
    """Retrieve configured sources; dry runs resolve remote listings without writing files."""
    dataset = config["dataset"]
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
        if dry_run:
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
    return {"dataset": "isic2016", "raw_root": str(root.resolve()),
            "archives": links.links, "dry_run": dry_run}
