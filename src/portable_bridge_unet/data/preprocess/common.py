"""Shared artifact writing for prepared binary segmentation datasets."""
import csv
import hashlib
import json
from pathlib import Path
import cv2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def output_root(config):
    raw = Path(config['dataset']['raw_root']).resolve()
    out = Path(config['dataset']['prepared_root']).resolve()
    if out == raw or raw in out.parents:
        raise ValueError('prepared_root must be outside raw_root')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty prepared_root; retain existing experiment manifests')
    for folder in ('images', 'masks', 'metadata'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    return out


def write_png(path, array):
    if not cv2.imwrite(str(path), array):
        raise OSError(f'Could not write {path}')


def write_metadata(out, rows, report):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / 'metadata/manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for split in ('train', 'val', 'test'):
        with (out / f'metadata/{split}_split.csv').open('w', newline='') as stream:
            keys = ['image', 'mask', 'group'] + (['roi'] if 'roi' in fields else [])
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(row for row in rows if row['split'] == split)
    report['prepared_root'] = str(out)
    (out / 'metadata/preparation.json').write_text(json.dumps(report, indent=2))
    return report
